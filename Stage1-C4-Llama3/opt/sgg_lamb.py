import torch
from torch.optim import Optimizer
from collections import defaultdict
from typing import Tuple, Optional, Callable, Dict, DefaultDict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math


class SGGLAMB(Optimizer):
    """Memory-optimized SGGLamb with efficient clustering integration, based on LAMB optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        n_clusters: int = 3,
        scale_update_freq: int = 500,
        scale_bound: Tuple[float, float] = (1, 10.0),
        beta3: float = 0.9,
        **kwargs,
    ):
        # Validate input parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError(f"n_clusters must be positive integer, got {n_clusters}")
        if not isinstance(scale_update_freq, int) or scale_update_freq < 1:
            raise ValueError(
                f"scale_update_freq must be positive integer, got {scale_update_freq}"
            )
        if len(scale_bound) != 2 or scale_bound[0] >= scale_bound[1]:
            raise ValueError(
                f"scale_bound must be (min, max) with min < max, got {scale_bound}"
            )
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"beta3 must be in [0, 1), got {beta3}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            n_clusters=n_clusters,
            scale_update_freq=scale_update_freq,
            scale_bound=scale_bound,
            beta3=beta3,
        )
        super().__init__(params, defaults)

        self.global_step = 0
        self.global_median = None

        # Memory-efficient cluster models
        self.cluster_models: DefaultDict[
            torch.nn.Parameter, MiniBatchKMeans
        ] = defaultdict(
            lambda: MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=128,
                compute_labels=False,
                max_no_improvement=5
            )
        )

        # Pinned memory buffers for efficient transfers
        self.pinned_buffers: Dict[torch.nn.Parameter, Optional[torch.Tensor]] = {}

        # Group-wise scaling factors with EMA smoothing
        self.group_scales: DefaultDict[int, float] = defaultdict(lambda: 1.0)

    def _align_gradients(
        self, param: torch.nn.Parameter, grad: torch.Tensor
    ) -> torch.Tensor:
        """Memory-efficient gradient alignment."""
        param_norm = param.data.norm(2)
        grad_norm = grad.norm(2)
        ratio = (param_norm + 1e-8) / (grad_norm + 1e-16)
        alignment_factor = torch.clamp(torch.log1p(ratio), 0.1, 10)
        return grad * alignment_factor

    def _compute_scale_factors(self) -> None:
        """Memory-efficient computation of group-wise scale factors."""
        param_map = {
            id(p): p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        if not param_map:
            return

        # Process groups one at a time to reduce memory
        for group_idx, group in enumerate(self.param_groups):
            group_grads = []
            for p in group["params"]:
                if p.grad is None or p.dim() <= 1:
                    continue
                grad = self._align_gradients(p, p.grad)
                group_grads.append(grad.view(-1))

            if not group_grads:
                continue

            # Compute group statistics
            group_grads_cat = torch.cat(group_grads)
            group_center = group_grads_cat.mean()
            group_distances = (group_grads_cat - group_center).abs()
            group_median_distance = max(
                torch.median(group_distances), self.defaults["eps"]
            )

            # Compute global statistics if not set
            if self.global_median is None:
                global_grads = []
                for other_group in self.param_groups:
                    for p in other_group["params"]:
                        if p.grad is not None and p.dim() > 1:
                            # param_size = torch.tensor(p.numel(), dtype=torch.float32, device=p.grad.device)
                            # normalized_grad = p.grad / torch.sqrt(param_size + 1e-16)  # sqrt norm
                            # global_grads.append(self._align_gradients(p, normalized_grad).view(-1))
                            global_grads.append(self._align_gradients(p, p.grad).view(-1))
                if global_grads:
                    global_grads_cat = torch.cat(global_grads)
                    global_center = global_grads_cat.mean()
                    global_distances = (global_grads_cat - global_center).abs()
                    self.global_median = max(
                        torch.median(global_distances), self.defaults["eps"]
                    ).item()

            if self.global_median is not None:
                # Compute scale with EMA
                group_scale = self.global_median / group_median_distance
                scale_adjustment = torch.log1p(
                    group_distances / group_median_distance
                ).mean()
                combined_scale = group_scale * scale_adjustment.item()

                # Clamp and update with EMA
                min_scale, max_scale = self.defaults["scale_bound"]
                clamped_scale = max(min_scale, min(max_scale, combined_scale))
                beta3 = self.defaults["beta3"]
                self.group_scales[group_idx] = (
                    beta3 * self.group_scales[group_idx] + (1 - beta3) * clamped_scale
                )

    def _update_clusters_and_scales(
        self, param: torch.nn.Parameter, state: dict, group: dict
    ) -> None:
        """Memory-optimized cluster update."""
        exp_avg_abs = state["exp_avg"].abs()
        
        # Initialize pinned buffer if needed
        if param not in self.pinned_buffers or self.pinned_buffers[param] is None:
            self.pinned_buffers[param] = torch.empty(
                exp_avg_abs.numel(), 
                dtype=torch.float32,
                pin_memory=True
            )
        
        # Copy data to pinned memory
        buffer = self.pinned_buffers[param][:exp_avg_abs.numel()]
        buffer.copy_(exp_avg_abs.flatten(), non_blocking=True)
        torch.cuda.synchronize()
        
        # Convert to numpy and cluster in batches
        flat_feat = buffer.cpu().numpy().reshape(-1, 1)
        km = self.cluster_models[param]
        km.partial_fit(flat_feat)
        
        # Predict clusters in batches to save memory
        batch_size = min(1024 * 1024, exp_avg_abs.numel())
        clusters = np.empty(exp_avg_abs.numel(), dtype=np.int32)
        
        for i in range(0, exp_avg_abs.numel(), batch_size):
            batch = flat_feat[i:i + batch_size]
            clusters[i:i + batch_size] = km.predict(batch)
        
        # Store clusters and compute scales
        state["clusters"] = torch.from_numpy(clusters).to(param.device)
        cluster_centers = torch.from_numpy(km.cluster_centers_.squeeze()).to(param.device)
        
        if self.global_median is not None:
            group_idx = self.param_groups.index(group)
            group_scale = self.group_scales[group_idx]
            group_median = exp_avg_abs.flatten().median()
            
            # Compute scales efficiently
            scales = (cluster_centers + group["eps"]) / (group_median + group["eps"])
            scales = scales * group_scale
            scale_mean = scales.mean()
            scales = scales / (scale_mean + group["eps"])
            
            # Apply per-cluster adjustments
            for i in range(len(scales)):
                mask = state["clusters"] == i
                if not mask.any():
                    continue
                    
                cluster_grads = exp_avg_abs.flatten()[mask]
                cluster_center = cluster_grads.mean()
                cluster_distances = (cluster_grads - cluster_center).abs()
                cluster_median_distance = max(
                    torch.median(cluster_distances), 
                    group["eps"]
                )
                scale_adjustment = torch.log1p(
                    cluster_distances / cluster_median_distance
                ).mean()
                scales[i] *= scale_adjustment.item()
            
            state["cluster_scale"] = scales.clamp_(*group["scale_bound"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Optimized step with memory-efficient clustering and LAMB update rule."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1
        self._compute_scale_factors()

        for group in self.param_groups:
            group_idx = self.param_groups.index(group)
            group_scale = self.group_scales[group_idx]

            for param in group["params"]:
                if param.grad is None:
                    continue

                # Optimize memory format
                if param.dim() >= 4 and not param.is_contiguous(memory_format=torch.channels_last):
                    param.data = param.data.contiguous(memory_format=torch.channels_last)
                    state = self.state[param]
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"].contiguous(memory_format=torch.channels_last)
                        state["exp_avg_sq"] = state["exp_avg_sq"].contiguous(memory_format=torch.channels_last)

                grad = self._align_gradients(param, param.grad)

                state = self.state[param]

                # Initialize state if needed
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["cluster_scale"] = torch.ones(group["n_clusters"], device=param.device)
                    state["clusters"] = None

                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Update moments
                exp_avg.mul_(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])
                exp_avg_sq.mul_(group["betas"][1]).addcmul_(grad, grad, value=1 - group["betas"][1])

                # Perform memory-efficient clustering
                if step % group["scale_update_freq"] == 0 and param.dim() > 1:
                    torch.cuda.empty_cache()
                    self._update_clusters_and_scales(param, state, group)
                    # Release buffer memory
                    if param in self.pinned_buffers:
                        self.pinned_buffers[param] = None

                # Compute bias corrections
                bias_correction1 = 1 - group["betas"][0] ** step
                bias_correction2 = 1 - group["betas"][1] ** step

                # LAMB update rule
                update = exp_avg / (exp_avg_sq.sqrt().add_(group["eps"]))
                if group["weight_decay"] != 0:
                    update.add_(param, alpha=group["weight_decay"])

                # Compute trust ratio
                param_norm = param.norm(2)
                update_norm = update.norm(2)
                trust_ratio = param_norm / (update_norm + group["eps"])
                trust_ratio = trust_ratio.clamp(*group["scale_bound"])

                # Compute base step size
                base_step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Apply scaling
                if param.dim() > 1 and "clusters" in state and state["clusters"] is not None:
                    scales = torch.index_select(
                        state["cluster_scale"], 0, state["clusters"].flatten()
                    ).view_as(update)
                    scaled_lr = base_step_size * group_scale * scales * trust_ratio
                    update.mul_(-scaled_lr)
                else:
                    update.mul_(-base_step_size * group_scale * trust_ratio)

                # Apply update
                param.add_(update)

        return loss
