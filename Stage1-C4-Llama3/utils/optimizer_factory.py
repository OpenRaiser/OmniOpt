import torch
import transformers
from loguru import logger

# 延迟导入bitsandbytes，避免在某些环境中的导入错误
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    BITSANDBYTES_AVAILABLE = False
    # 创建一个假的bnb模块以避免后续导入错误
    class FakeBNB:
        class optim:
            class Adam8bit:
                pass
    bnb = FakeBNB()

from opt import (
    GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, APOLLOAdamW, QAPOLLOAdamW,
    AdaBelief, Adam_mini, AdamP, Adan, Adopt, CAME, CondaAdamW, Kron, Lamb, LARS, Lion,
    LaProp, MARS, Muon, RMNP, NvNovoGrad, Prodigy, Shampoo, SOAP, SophiaG,
    SGGAdamW, SGGLAMB, SGGShampoo,
    AdamWLegacy, NAdamLegacy, RAdamLegacy,
)

from .training_utils import get_scheculer


def setup_optimization(args, model, trainable_params, param_groups, id_lowrank_params, model_config=None):
    """
    Setup optimizer and scheduler based on the specified optimizer type.

    Args:
        args: Command line arguments
        model: The model to optimize
        trainable_params: List of trainable parameters
        param_groups: Parameter groups for low-rank optimizers
        id_lowrank_params: IDs of low-rank parameters
        model_config: Model configuration for extracting dim and n_heads

    Returns:
        model, optimizer, scheduler, layer_wise_flag
    """
    layer_wise_flag = False

    # Standard optimizers
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=getattr(args, 'beta1', 0.9)
        )

    # AdaBelief optimizer
    elif args.optimizer.lower() == "adabelief":
        optimizer = AdaBelief(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-12 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-12)
        )

    # Adam_mini optimizer
    elif args.optimizer.lower() == "adam_mini":
        trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        optimizer = Adam_mini(
            trainable_named_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8),
            dim=model_config.hidden_size if model_config else getattr(args, 'dim', 4096),
            n_heads=model_config.num_attention_heads if model_config else getattr(args, 'n_heads', 32)
        )

    # AdamP optimizer
    elif args.optimizer.lower() == "adamp":
        optimizer = AdamP(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            delta=0.1,
            wd_ratio=0.1,
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Adan optimizer
    elif args.optimizer.lower() == "adan":
        optimizer = Adan(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.98, 0.92, 0.99) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2, args.beta3),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Adopt optimizer
    elif args.optimizer.lower() == "adopt":
        optimizer = Adopt(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.9999) if getattr(args, 'beta1', None) is None else (0.9, args.beta2),
            eps=1e-6 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-6)
        )

    # CAME optimizer (Confidence-guided Adaptive Memory Efficient)
    # Uses 3 betas: (update_ema, sq_grad_ema, instability_ema)
    # Uses eps tuple: (sq_grad_reg, instability_reg)
    # Paper recommends: betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16)
    elif args.optimizer.lower() == "came":
        came_beta1 = getattr(args, 'beta1', 0.9)
        came_beta2 = args.beta2 if hasattr(args, 'beta2') else 0.999
        came_eps_val = getattr(args, 'eps', None)
        if came_eps_val is not None:
            came_eps = (1e-30, came_eps_val)
        else:
            came_eps = (1e-30, 1e-16)
        optimizer = CAME(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(came_beta1, came_beta2, 0.9999),
            eps=came_eps,
            clip_threshold=1.0,
        )

    # Kron optimizer
    elif args.optimizer.lower() == "kron":
        optimizer = Kron(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9 if getattr(args, 'beta1', None) is None else getattr(args, 'beta1', 0.9),
            memory_save_mode="one_diag"
        )

    # LAMB optimizer
    elif args.optimizer.lower() == "lamb":
        optimizer = Lamb(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-6 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-6)
        )

    # LARS optimizer
    elif args.optimizer.lower() == "lars":
        optimizer = LARS(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9 if getattr(args, 'beta1', None) is None else getattr(args, 'beta1', 0.9),
            nesterov=True
        )

    # Lion optimizer
    elif args.optimizer.lower() == "lion":
        optimizer = Lion(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2)
        )

    # LaProp optimizer
    elif args.optimizer.lower() == "laprop":
        optimizer = LaProp(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-15 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-15)
        )

    # MARS optimizer variants
    elif "mars" in args.optimizer.lower():
        mars_type = {
            "mars": "mars-adamw",
            "mars_adamw": "mars-adamw",
            "mars_lion": "mars-lion",
            "mars_shampoo": "mars-shampoo"
        }
        optimizer = MARS(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.99) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
            gamma=0.025,
            is_approx=True,
            mars_type=mars_type[args.optimizer.lower()],
            optimize_1d=False,
            lr_1d=args.lr,
            betas_1d=(0.9, 0.95),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Muon optimizer
    elif args.optimizer.lower() == "muon":
        trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        optimizer = Muon(
            trainable_named_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.95,
            betas=(0.9, 0.95) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
            nesterov=True if getattr(args, 'nesterov', None) is None else getattr(args, 'nesterov', True),
            ns_steps=5 if getattr(args, 'ns_steps', None) is None else getattr(args, 'ns_steps', 5),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # RMNP optimizer (like Muon: RMNP update for 2D+ params, Adam for 1D/0D)
    elif args.optimizer.lower() == "rmnp":
        rmnp_params = [p for name, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
        non_rmnp_params = [p for name, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
        optimizer = RMNP(
            rmnp_params,
            lr=args.lr,
            rmnp_params=rmnp_params,
            adam_params=non_rmnp_params,
            lr_adam=args.lr if getattr(args, 'adam_lr', None) is None else args.adam_lr,
            weight_decay=args.weight_decay,
            momentum=0.95 if getattr(args, 'beta1', None) is None else args.beta1,
            beta=0.95 if getattr(args, 'beta2', None) is None else args.beta2,
            betas=(
                0.9 if getattr(args, 'adam_beta1', None) is None else args.adam_beta1,
                0.999 if getattr(args, 'adam_beta2', None) is None else args.adam_beta2,
            ),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8),
        )
    # NAdam optimizer
    elif args.optimizer.lower() == "nadam":
        optimizer = torch.optim.NAdam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # NvNovoGrad optimizer
    elif args.optimizer.lower() == "novograd":
        optimizer = NvNovoGrad(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.98) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Prodigy optimizer
    elif args.optimizer.lower() == "prodigy":
        optimizer = Prodigy(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # RAdam optimizer
    elif args.optimizer.lower() == "radam":
        optimizer = torch.optim.RAdam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Shampoo optimizer
    elif args.optimizer.lower() == "shampoo":
        optimizer = Shampoo(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # SOAP optimizer
    elif args.optimizer.lower() == "soap":
        optimizer = SOAP(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.95) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
            precondition_frequency=10,
            precondition_1d=False,
            normalize_grads=False,
            data_format="channels_last",
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # SophiaG optimizer
    elif args.optimizer.lower() == "sophia":
        optimizer = SophiaG(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.965, 0.99) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2)
        )

    # SGG optimizers
    elif args.optimizer.lower() in ["sgg_adamw", "sggadamw"]:
        optimizer = SGGAdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
            scale_bound=(args.scale_bound if args.scale_bound is not None else 0.5, 10.0),
            n_clusters=getattr(args, 'n_clusters', 3),
            scale_update_freq=getattr(args, 'scale_update_freq', 500),
            beta3=getattr(args, 'beta3', 0.9),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() in ["sgg_lamb", "sgglamb"]:
        optimizer = SGGLAMB(
            trainable_params,
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
            scale_bound=(args.scale_bound if args.scale_bound is not None else 0.5, 10.0),
            n_clusters=getattr(args, 'n_clusters', 3),
            scale_update_freq=getattr(args, 'scale_update_freq', 500),
            beta3=getattr(args, 'beta3', 0.9),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() in ["sgg_shampoo", "sggshampoo"]:
        optimizer = SGGShampoo(
            trainable_params,
            lr=args.lr,
            betas=(0.95, args.beta2),
            weight_decay=args.weight_decay,
            scale_bound=(args.scale_bound if args.scale_bound is not None else 0.5, 10.0),
            n_clusters=getattr(args, 'n_clusters', 3),
            scale_update_freq=getattr(args, 'scale_update_freq', 500),
            beta3=getattr(args, 'beta3', 0.9),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # GaLore optimizers
    elif args.optimizer.lower() == "galore_adamw":
        optimizer = GaLoreAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() == "apollo_adamw":
        optimizer = APOLLOAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            scale_front=args.scale_front,
            eps=1e-6 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-6)
        )
    elif args.optimizer.lower() == "q_apollo":
        if QAPOLLOAdamW is None:
            raise ImportError("QAPOLLOAdamW requires bitsandbytes which is not available. Please install bitsandbytes or use a different optimizer.")
        optimizer = QAPOLLOAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
            scale_front=args.scale_front,
        )
    elif args.optimizer.lower() in ["conda", "conda_adamw", "condaadamw"]:
        optimizer = CondaAdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            scale_front=args.scale_front,
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )

    # Adafactor optimizer
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if getattr(args, 'beta1', 0.0) == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None if getattr(args, 'beta1', None) is None else args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if getattr(args, 'beta1', 0.0) == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None if getattr(args, 'beta1', None) is None else args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    # 8-bit optimizers
    elif args.optimizer.lower() == "adam8bit":
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is not available. Cannot use 8-bit optimizers.")
        optimizer = bnb.optim.Adam8bit(
            trainable_params,
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() == "galore_adamw8bit":
        if GaLoreAdamW8bit is None:
            raise ImportError("GaLoreAdamW8bit requires bitsandbytes which is not available. Please install bitsandbytes or use a different optimizer.")
        optimizer = GaLoreAdamW8bit(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, args.beta2),
            eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
        )
    elif args.optimizer.lower() == "q_galore_adamw8bit":
        if getattr(args, 'simulation', False):
            print("Using Simulation Mode")
            optimizer = QGaLoreAdamW8bit_simulate(
                param_groups, lr=args.lr, weight_decay=args.weight_decay,
                betas=(0.9, args.beta2) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
                eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
            )
        else:
            optimizer = QGaLoreAdamW8bit(
                param_groups, lr=args.lr, weight_decay=args.weight_decay,
                betas=(0.9, args.beta2) if getattr(args, 'beta1', None) is None else (args.beta1, args.beta2),
               eps=1e-8 if getattr(args, 'eps', None) is None else getattr(args, 'eps', 1e-8)
            )

    # Layer-wise optimizers
    elif args.optimizer.lower() == "galore_adamw8bit_per_layer":
        if GaLoreAdamW8bit is None:
            raise ImportError("GaLoreAdamW8bit requires bitsandbytes which is not available. Please install bitsandbytes or use a different optimizer.")
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_lowrank_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": args.rank,
                                "update_proj_gap": args.update_proj_gap * 2,
                                "scale": args.galore_scale,
                                "proj_type": args.proj_type,
                            }
                        ],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    if not BITSANDBYTES_AVAILABLE:
                        raise ImportError("bitsandbytes is not available. Cannot use 8-bit optimizers.")
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    elif args.optimizer.lower() == "q_galore_adamw8bit_per_layer":
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params:
                optimizer_dict[p] = QGaLoreAdamW8bit(
                    [
                        {
                            "params": [p],
                            "rank": args.rank,
                            "update_proj_gap": args.update_proj_gap * 2,
                            "scale": args.galore_scale,
                            "proj_type": args.proj_type,
                            "quant": args.proj_quant,
                            "quant_n_bit": args.proj_bits,
                            "quant_group_size": args.proj_group_size,
                            "cos_threshold": args.cos_threshold,
                            "gamma_proj": args.gamma_proj,
                            "queue_size": args.queue_size,
                        }
                    ],
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
            else:
                if p.requires_grad:
                    if not BITSANDBYTES_AVAILABLE:
                        raise ImportError("bitsandbytes is not available. Cannot use 8-bit optimizers.")
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params or p.requires_grad:
                scheduler_dict[p] = get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if (not hasattr(p, "float_grad")) and p.grad is None:
                return

            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        layer_wise_flag = True

    elif args.optimizer.lower() == "q_apollo_per_layer":
        if QAPOLLOAdamW is None:
            raise ImportError("QAPOLLOAdamW requires bitsandbytes which is not available. Please install bitsandbytes or use a different optimizer.")
        optimizer_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params:
                optimizer_dict[p] = QAPOLLOAdamW(
                    [
                        {
                            "params": [p],
                            "rank": args.rank,
                            "update_proj_gap": args.update_proj_gap,
                            "scale": args.apollo_scale,
                            "proj_type": args.proj_type,
                            "proj": args.proj,
                            "scale_type": args.scale_type,
                        }
                    ],
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
            else:
                if p.requires_grad:
                    if not BITSANDBYTES_AVAILABLE:
                        raise ImportError("bitsandbytes is not available. Cannot use 8-bit optimizers.")
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if id(p) in id_lowrank_params or p.requires_grad:
                scheduler_dict[p] = get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if (not hasattr(p, "float_grad")) and p.grad is None:
                return

            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        # return the dict instead
        optimizer = optimizer_dict
        scheduler = scheduler_dict

    return model, optimizer, scheduler, layer_wise_flag
