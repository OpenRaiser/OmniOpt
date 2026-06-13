import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
import torch.distributed as dist

from transformers.utils.versions import require_version


class CondaProjector:
    def __init__(self, verbose=False, update_proj_gap=2000, scale=1.0, proj_type='std'):
        """
        Args:
            verbose (bool): Whether to print debug information.
            update_proj_gap (int): How often (in steps) to update the orthogonal matrix.
            scale (float): Scale factor to apply when projecting back.
            proj_type (str): Projection type ('std', 'reverse_std', 'right', 'left', or 'full').
        """
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.last_svd_step = -1  # Step at which SVD was last performed

    def state_dict(self):
        ortho = self.ortho_matrix
        if isinstance(ortho, (list, tuple)):
            # full proj_type: [U, Vh]; ortho_type=2
            return {
                "ortho_type": 2,
                "ortho_0": ortho[0].cpu() if ortho[0] is not None else torch.tensor([]),
                "ortho_1": ortho[1].cpu() if ortho[1] is not None else torch.tensor([]),
                "last_svd_step": self.last_svd_step,
            }
        elif isinstance(ortho, torch.Tensor):
            # ortho_type=1
            return {
                "ortho_type": 1,
                "ortho_0": ortho.cpu(),
                "last_svd_step": self.last_svd_step,
            }
        else:
            # ortho_type=0: not yet computed
            return {
                "ortho_type": 0,
                "last_svd_step": self.last_svd_step,
            }

    def load_state_dict(self, state):
        ortho_type = state["ortho_type"]
        if ortho_type == 2:
            self.ortho_matrix = [state["ortho_0"], state["ortho_1"]]
        elif ortho_type == 1:
            self.ortho_matrix = state["ortho_0"]
        else:
            self.ortho_matrix = None
        self.last_svd_step = state["last_svd_step"]

    def project_with_cached_ortho(self, input_matrix, svd_basis_matrix, step):
        if input_matrix.dim() < 2:
            return input_matrix

        update_condition = self.ortho_matrix is None or step % self.update_proj_gap == 0
        already_updated_this_step = step == self.last_svd_step

        # Only update the orthogonal matrix if necessary and not already updated at this step
        if update_condition and not already_updated_this_step:
            if self.proj_type == 'std':
                if input_matrix.shape[0] >= input_matrix.shape[1]:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
                else:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
            elif self.proj_type == 'reverse_std':
                if input_matrix.shape[0] >= input_matrix.shape[1]:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
                else:
                    self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
            elif self.proj_type == 'right':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='right')
            elif self.proj_type == 'left':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='left')
            elif self.proj_type == 'full':
                self.ortho_matrix = self.get_orthogonal_matrix(svd_basis_matrix, type='full')
            else:
                raise ValueError(f"Unknown proj_type: {self.proj_type}")
            self.last_svd_step = step

        return self.project(input_matrix, svd_basis_matrix, step, cached_ortho_matrix=self.ortho_matrix)

    def project(self, input_matrix, svd_basis_matrix, step, cached_ortho_matrix=None):

        if cached_ortho_matrix is not None:
            self.ortho_matrix = cached_ortho_matrix

        device = input_matrix.device
        ortho = self.ortho_matrix

        if self.proj_type == 'std':
            if input_matrix.shape[0] >= input_matrix.shape[1]:
                projected_matrix = torch.matmul(input_matrix, ortho.transpose(-2, -1).to(device))
            else:
                projected_matrix = torch.matmul(ortho.transpose(-2, -1).to(device), input_matrix)
        elif self.proj_type == 'reverse_std':
            if input_matrix.shape[0] >= input_matrix.shape[1]:
                projected_matrix = torch.matmul(ortho.transpose(-2, -1).to(device), input_matrix)
            else:
                projected_matrix = torch.matmul(input_matrix, ortho.transpose(-2, -1).to(device))
        elif self.proj_type == 'right':
            projected_matrix = torch.matmul(input_matrix, ortho.transpose(-2, -1).to(device))
        elif self.proj_type == 'left':
            projected_matrix = torch.matmul(ortho.transpose(-2, -1).to(device), input_matrix)
        elif self.proj_type == 'full':
            # ortho is a list of [U, Vh]
            left, right = ortho
            projected_matrix = torch.matmul(left.transpose(-2, -1).to(device), input_matrix)
            projected_matrix = torch.matmul(projected_matrix, right.transpose(-2, -1).to(device))
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return projected_matrix

    def project_back(self, projected_matrix):
        if projected_matrix.dim() < 2:
            return projected_matrix

        device = projected_matrix.device
        ortho = self.ortho_matrix

        if self.proj_type == 'std':
            if projected_matrix.shape[0] >= projected_matrix.shape[1]:
                projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
            else:
                projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
        elif self.proj_type == 'reverse_std':
            if projected_matrix.shape[0] <= projected_matrix.shape[1]:
                projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
            else:
                projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
        elif self.proj_type == 'right':
            projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
        elif self.proj_type == 'left':
            projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
        elif self.proj_type == 'full':
            # ortho is a list of [U, Vh]
            left, right = ortho
            projected_back_matrix = torch.matmul(left.to(device), projected_matrix)
            projected_back_matrix = torch.matmul(projected_back_matrix, right.to(device))
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return projected_back_matrix * self.scale

    def get_orthogonal_matrix(self, svd_basis_matrix, type):
        """
        Compute the orthogonal matrix (U or Vh, or both) via SVD.

        Args:
            svd_basis_matrix (Tensor): Input matrix for SVD.
            type (str): 'left', 'right', or 'full'.

        Returns:
            Tensor or list: U, Vh, or [U, Vh] from SVD.
        """
        matrix = svd_basis_matrix.data
        orig_dtype = matrix.dtype
        orig_device = matrix.device

        # Perform SVD in float32 for numerical stability 
        if orig_dtype != torch.float:
            matrix = matrix.float()

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if type == 'right':
            B = Vh
            if orig_dtype != torch.float:
                B = B.to(orig_device).type(orig_dtype)
            return B
        elif type == 'left':
            A = U
            if orig_dtype != torch.float:
                A = A.to(orig_device).type(orig_dtype)
            return A
        elif type == 'full':
            A = U
            B = Vh
            if orig_dtype != torch.float:
                A = A.to(orig_device).type(orig_dtype)
                B = B.to(orig_device).type(orig_dtype)
            return [A, B]
        else:
            raise ValueError("type should be 'left', 'right', or 'full'")


class Conda(Optimizer):
    """
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        update_proj_gap: int = 2000,
        scale: float = 0.25,
        proj_type: str = "std",
        rank: int = 256,
        **kwargs,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "correct_bias": correct_bias, "update_proj_gap": update_proj_gap,
            "scale": scale, "proj_type": proj_type, "rank": rank,
        }
        super().__init__(params, defaults)

    def state_dict(self):
        sd = super().state_dict()
        # Flatten CondaProjector into top-level tensor/scalar keys so DCP can serialize them
        for state in sd["state"].values():
            if "projector" in state:
                proj = state.pop("projector")
                proj_sd = proj.state_dict()
                for k, v in proj_sd.items():
                    state[f"__proj_{k}"] = v
        return sd

    def load_state_dict(self, state_dict):
        # Extract projector flat keys before super() processes the state
        projector_states = {}
        for param_id, state in state_dict["state"].items():
            proj_keys = [k for k in state if k.startswith("__proj_")]
            if proj_keys:
                proj_sd = {k[len("__proj_"):]: state.pop(k) for k in proj_keys}
                projector_states[param_id] = proj_sd

        super().load_state_dict(state_dict)

        all_params = [p for group in self.param_groups for p in group["params"]]
        for param_id, proj_sd in projector_states.items():
            idx = int(param_id)
            if idx < len(all_params):
                p = all_params[idx]
                live_state = self.state.get(p, {})
                if "projector" not in live_state:
                    live_state["projector"] = CondaProjector()
                    self.state[p] = live_state
                live_state["projector"].load_state_dict(proj_sd)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                
                # Conda Projection — only for 2D params; 3D+ (e.g. deltanet state) unsupported
                if "update_proj_gap" in group and grad.dim() == 2:
                    if "projector" not in state:
                        state["projector"] = CondaProjector(update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project_with_cached_ortho(grad, exp_avg, state["step"])
                    exp_avg = state["projector"].project_with_cached_ortho(exp_avg, exp_avg, state["step"])
                     
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                state["step"] += 1

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # Projection Back
                if "update_proj_gap" in group and "projector" in state:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                
                
        return loss
