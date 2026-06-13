import os
import json
import torch
import argparse
from loguru import logger
from datetime import datetime


def parse_args(args):
    parser = argparse.ArgumentParser()

    ### Experiment setup ###
    parser.add_argument("--model_config", type=str, default="configs/llama_350m.json")
    parser.add_argument("--exp_config", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", default=True, action="store_true")
    parser.add_argument("--dynamo_suppress_errors", default=True, action="store_true",
                       help="Suppress TorchDynamo errors and fall back to eager mode")
    parser.add_argument("--dynamo_cache_limit", type=int, default=10000,
                       help="TorchDynamo cache size limit to prevent memory issues")
    parser.add_argument("--memory_cleanup_frequency", type=int, default=10000,
                       help="Frequency of memory cleanup during training (in steps)")

    parser.add_argument("--resume_step", type=int, default=None) # if None, resume from the latest checkpoint
    parser.add_argument("--restore_optimizer", action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)

    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--workers", type=int, default=12)

    ### Training hyperparameters ###
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_lr", type=float, default=None, help="Separate AdamW lr for RMNP's 1D params; defaults to --lr if not set")
    parser.add_argument("--adam_beta1", type=float, default=None, help="Adam beta1 for RMNP's Adam branch (1D/0D params); default 0.9")
    parser.add_argument("--adam_beta2", type=float, default=None, help="Adam beta2 for RMNP's Adam branch (1D/0D params); default 0.999")
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                            "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                            "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--activation_checkpointing", action="store_true")

    ### Dataset hyperparameters ###
    parser.add_argument("--data_path", type=str, default="c4", help="Dataset path/name to load")
    parser.add_argument("--data_name", type=str, default="en", help="Dataset subset/configuration name")

    ### Wandb ###
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--project", type=str, default="test")
    parser.add_argument("--unset_wandb", action="store_true")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None, 
                        help="Directory to save wandb logs in offline mode. If not specified, will use save_dir")

    ### Adaptive optimization hyperparameters ###
    parser.add_argument("--beta1", type=float, default=0.9) # beta1 for Adafactor, GaLore_adafactor, (Q-)GaLore-adam or SGD
    parser.add_argument("--beta2", type=float, default=0.999) # beta2 for (Q-)GaLore-adam
    parser.add_argument("--beta3", type=float, default=0.99) # beta3 for Adan
    parser.add_argument("--eps", type=float, default=None) # epsilon for various optimizers
    # GaLore hyperparameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    # Q-GaLore hyperparameters: quantization
    parser.add_argument("--proj_quant", action='store_true')
    parser.add_argument("--proj_bits", type=int, default=8)
    parser.add_argument("--proj_group_size", type=int, default=256)
    parser.add_argument("--weight_quant", action='store_true')
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--weight_group_size", type=int, default=256)
    parser.add_argument("--stochastic_round", action='store_true')
    parser.add_argument("--simulation", action='store_true')
    parser.add_argument("--cos_threshold", type=float, default=1)
    parser.add_argument("--gamma_proj", type=int, default=2)
    parser.add_argument("--queue_size", type=int, default=5)
    # APOLLO hyperparameters
    parser.add_argument("--proj", type=str, default="random") # "random" or "svd"
    parser.add_argument("--scale_type", type=str, default="tensor") # "tensor" or "channel"
    parser.add_argument("--scale_front", action='store_true') # put the nl before or after scale the gradient with the apollo_scale
    
    # SGG hyperparameters
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--scale_update_freq", type=int, default=500)
    parser.add_argument("--scale_bound", type=float, default=None)
    parser.add_argument("--metric", type=str, default="mean")
    
    # Additional optimizer parameters
    parser.add_argument("--dim", type=int, default=4096) # for Adam_mini
    parser.add_argument("--n_heads", type=int, default=32) # for Adam_mini
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_momentum", type=float, default=0.95)

    args = parser.parse_args(args)
    # Setup experiments
    if args.exp_config is not None:
        with open(args.exp_config, "r") as f:
            exp_config = json.load(f)
        args.__dict__.update(exp_config)
    args = check_args_torchrun_main(args)
    return args


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)


def check_args_torchrun_main(args):

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    os.makedirs(args.save_dir, exist_ok=True)

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    return args
