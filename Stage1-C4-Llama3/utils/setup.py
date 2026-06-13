import torch
import random
import numpy as np
from loguru import logger

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from .modeling_llama import LlamaForCausalLM

from .fake_quantization import QLinear, prepare_model_for_int8_training_simulation
from .quantization import QScaleLinear, prepare_model_for_int8_training


def getting_svd_cnt(optimizer):
    svd_cnt = 0
    state = optimizer.state_dict().get("state", {})
    for key in state:
        if "projector" in state[key]:
            if hasattr(state[key]["projector"], "svd_count"):
                svd_cnt += state[key]["projector"].svd_count
    return svd_cnt


def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set seed for all GPUs
    torch.cuda.manual_seed_all(args.seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model(args):
    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.weight_quant:
        assert args.optimizer.lower() in [
            "q_galore_adamw8bit",
            "q_galore_adamw8bit_per_layer",
            "q_apollo",
            "q_apollo_per_layer",
        ]
        target_module = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        if args.simulation:
            model = prepare_model_for_int8_training_simulation(model, args, target_module)
        else:
            model = prepare_model_for_int8_training(model, args, target_module)
        logger.info("--" * 20)
        logger.info("Prepare Model for Int8 Training")
        logger.info("--" * 20)

    return model_config, model


def saving_model_weight(model, path, args):
    """
    Save model weight to file
    """
    checkpoint = model.state_dict()
    if args.simulation and args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QLinear):
                checkpoint[name + ".weight"] = module.weight
                if module.bias is not None:
                    checkpoint[name + ".bias"] = module.bias
                checkpoint[name + ".group_size"] = module.weight.group_size
                checkpoint[name + ".stochastic_round"] = module.weight.stochastic_round
                checkpoint[name + ".num_bits"] = module.num_bits
                checkpoint[name + ".group_size"] = module.group_size

    elif args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QScaleLinear):
                checkpoint[name + ".weight"] = module.weight
                if module.bias is not None:
                    checkpoint[name + ".bias"] = module.bias
                checkpoint[name + ".scales"] = module.weight.scales
                checkpoint[name + ".zeros"] = module.weight.zeros
                checkpoint[name + ".group_size"] = module.weight.group_size
                checkpoint[name + ".saved_data_dtype"] = module.weight.saved_data_dtype
                checkpoint[name + ".stochastic_round"] = module.weight.stochastic_round
    else:
        print("saving model weight without quantized layer")
    torch.save(checkpoint, path)


def load_model_weight(model, path, args):
    """
    Load model weight from file
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    if args.simulation and args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QLinear):
                module.weight = checkpoint[name + ".weight"]
                if module.bias is not None:
                    module.bias = checkpoint[name + ".bias"]
                module.weight.group_size = checkpoint[name + ".group_size"]
                module.weight.stochastic_round = checkpoint[name + ".stochastic_round"]
                module.num_bits = checkpoint[name + ".num_bits"]
                module.group_size = checkpoint[name + ".group_size"]

    elif args.weight_quant:
        for name, module in model.named_modules():
            if isinstance(module, QScaleLinear):
                module.weight = checkpoint[name + ".weight"]
                if module.bias is not None:
                    module.bias = checkpoint[name + ".bias"]
                module.weight.scales = checkpoint[name + ".scales"]
                module.weight.zeros = checkpoint[name + ".zeros"]
                module.weight.group_size = checkpoint[name + ".group_size"]
                module.weight.saved_data_dtype = checkpoint[name + ".saved_data_dtype"]
                module.weight.stochastic_round = checkpoint[name + ".stochastic_round"]
    else:
        print("loading model weight without quantized layer")
    return model
