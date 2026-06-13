import torch
import torch.distributed as dist
from typing import Iterable, Tuple, Union


# Aligned with the latest KellerJordan/Muon (https://github.com/KellerJordan/Muon/blob/master/muon.py)
@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Supports batched inputs (G.ndim >= 2).
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """
    Compute the Muon update for a single parameter.
    Uses lerp_ for momentum update (aligned with latest official implementation).

    LR scaling: max(1, rows/cols)^0.5
    ⚠️  IMPORTANT: This is a significant change from the old scaling `0.2 * sqrt(max(A,B))`.
    The effective update magnitude is now much smaller per unit lr.
    - Old: lr=1e-3 with old scaling → effective_lr ≈ 6.4e-3 for [1024,1024]
    - New: lr=1e-3 with new scaling → effective_lr = 1e-3 for [1024,1024]
    If switching from old to new, you should increase lr by ~6x (e.g., lr=6e-3 or use lr=0.02
    as recommended by the official implementation).
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim > 2:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Updated LR scaling: max(1, rows/cols)^0.5
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Aligned with the latest KellerJordan/Muon interface.
    Supports both regular params and named_parameters format.
    2D params (excluding embed/head) use Muon, others use AdamW.

    Key updates from previous version:
    - LR scaling changed from `0.2 * sqrt(max(A,B))` to `max(1, rows/cols)^0.5`
    - Momentum update uses `lerp_` instead of `mul_.add_`
    - Batched Newton-Schulz (supports G.ndim >= 2, uses mT instead of T)
    - Nesterov update uses `lerp_` for cleaner implementation

    Arguments:
        params: The parameters to be optimized. 2D params (excl. embed/head) use Muon, others use AdamW.
        lr: The learning rate. (0.02 is a good default for Muon; 1e-3 for AdamW backup)
        weight_decay: Weight decay for both Muon and AdamW.
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum. (recommended)
        ns_steps: The number of Newton-Schulz iterations. (5 is usually enough)
        betas: The betas for the internal AdamW backup.
        eps: The epsilon for the internal AdamW backup.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Tuple[str, torch.nn.Parameter]]],
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        betas=(0.9, 0.95),
        eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
        )

        # Check if params is named_parameters format (tuples of (name, param))
        if params and isinstance(next(iter(params)), tuple):
            param_list = [p for name, p in params]  # named_parameters to regular params
            super().__init__(param_list, defaults)

            # Define embedding and output layer names
            embd_names = {"embed", "embd", "wte", "embed_tokens"}
            output_names = {"lm_head", "output", "final_layer"}
            for param_name, param in params:
                param_name = param_name.lower()
                if not param.requires_grad:
                    continue
                self.state[param]["use_muon"] = (
                    param.ndim >= 2 and
                    not any(embd_name in param_name for embd_name in embd_names) and
                    not any(output_name in param_name for output_name in output_names)
                )
        else:
            super().__init__(params, defaults)
            # Sort parameters into those for which we will use Muon, and those for which we will not
            for group in self.param_groups:
                for p in group['params']:
                    # Use Muon for 2D parameters, AdamW for others
                    self.state[p]["use_muon"] = p.ndim == 2

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # Use muon_update aligned with latest official version
                update = muon_update(
                    g,
                    buf,
                    beta=momentum,
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )

                # Apply weight decay and update
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update.reshape(p.shape), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
