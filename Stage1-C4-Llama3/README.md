# 🚀 APOLLO: SGD-like Memory, AdamW-level Performance, MLSys'2025
APOLLO is archievd at Figshare with DOI: https://doi.org/10.6084/m9.figshare.28558319.v1 for MLSys artifact evaluation.

A memory-efficient optimizer designed for **large language model (LLM) pre-training** and **full-parameter fine-tuning**, offering **SGD-like memory cost** with **AdamW-level performance**.

## How to use?

### 📦 Installation

### Install basic conda env
```bash
conda create -n llama python=3.11 -y
conda activate llama
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -i https://mirrors.westlake.edu.cn/pypi/simple
# install nvcc (optional)
conda config --env --add channels nvidia
conda config --env --set channel_priority strict
conda install -y cuda-toolkit=12.4
## build flash-attn w/ or w/o compile
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Install APOLLO via pip
You can install the APOLLO optimizer directly from pip:
```bash
pip install apollo-torch
```

### Install experiment dependencies from source
To install SAC from the source code:

```bash
git clone https://github.com/ScalingOpt/SAC.git
cd SAC
pip install -r exp_requirements.txt
pip install -e .
```

### 📖 Usage

#### Save optimizer memory using APOLLO optimizers
```
from opt import APOLLOAdamW
# define param groups as lowrank_params and regular params
param_groups = [{'params': non_lowrank_params}, 
                {'params': 
                  lowrank_params, 
                  'rank': 1, 
                  'proj': 'random', 
                  'scale_type': 'tensor', 
                  'scale': 128,
                  'update_proj_gap': 200, 
                  'proj_type': 'std'}]
optimizer = APOLLO(param_groups, lr=0.01)
```

#### Hyperparameter choices
For APOLLO and APOLLO-Mini, we have the following arguments

#### `rank`
- Specifies the rank of the auxiliary sub-space used for gradient scaling.
- **Default value:** 
    - `256` for APOLLO works well for 1B and 7B model.
    - `1` for APOLLO-Mini. 

#### `scale_type`
- Determines how the scaling factors are applied:
  - **`channel`**: Applies gradient scaling at the channel level (APOLLO)
  - **`tensor`**: Applies gradient scaling at the tensor level (APOLLO-Mini).

#### **`scale`**
The `scale` parameter plays a crucial role in heuristically adjusting gradient updates to compensate for scaling factor approximation errors arising from the use of a lower rank. Proper tuning of this parameter can significantly improve performance:
- **`1`**: Default value for APOLLO (validated on A100 GPUs).
- **`128`**: Default value for APOLLO-Mini. For larger models, experimenting with higher values is recommended.

#### `--scale_front`

To stabilize training, we adopt the **Norm-Growth Limiter (NL)** from [Fira](https://github.com/xichen-fy/Fira), which has shown to be slightly more effective than traditional gradient clipping.

There are two ways to apply the Norm-Growth Limiter based on when it's used relative to the heuristical (`scale`):
1. **After Scaling**: NL is applied after the gradient is multiplied by the `scale`.
   - Recommended for when training involves fewer warmup steps, e.g., LLaMA 60M and 130M with APOLLO-Mini.
   - Enable this by setting `--scale_front`.
2. **Before Scaling**: NL is applied before the gradient is scaled.
   - With sufficient warmup steps, both methods yield similar performance for large models.

---

### Benchmark 1: Pre-Training LLaMA on C4 dataset

We provide the command in `scripts/benchmark_c4` for pretraining LLaMA model with sizes from 60M to 7B on C4 dataset.

```
# num_rank: 1 for APOLLO-Mini, 1/4 of the original dim for APOLLO (same as Galore)
# scale_type: channel or tensor
# projection type: random (option: svd)
# scale: related with rank, larger rank generally works well with smaller scale, we use 128 for rank=1

```

### Benchmark 2: Pre-Training LLaMA on C4 dataset with long context window
Compared to academic settings, the industry trains large language models (LLMs) with significantly longer context windows (1k-8k tokens) and on hundreds of billions of tokens. 

Accordingly, we further validate the effectiveness of the **APOLLO** series by pre-training a **LLaMA-350M** on a 1024-token context window—**four times larger than the original GaLore usage**. To establish a robust baseline, we vary **AdamW**’s learning rate across `[1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2]`. We also “lazily” tune the scale factor of the **APOLLO** series by testing **APOLLO** in `[√1, √2, √3]` and **APOLLO-Mini** in `[√128, √256, √384]`, while keeping the learning rate fixed at `1e-2`.

Both **APOLLO** and **APOLLO-Mini** demonstrate superior performance compared to **AdamW**, while drastically reducing optimizer memory usage—by as much as 1/8 or even 1/1024 of AdamW’s requirements. Moreover, these methods tend to exhibit even stronger performance in later stages, when more training tokens are involved. This makes them a highly promising option for partial LLM pre-training scenarios involving long context windows and trillions of training tokens.

<div align="center">
  <img src="https://raw.githubusercontent.com/zhuhanqing/APOLLO/main/docs/static/images/apollo_350m_long_context.jpg" alt="APOLLO 350M long context" width="80%">
</div>

*Figure 3:  Perplexity curves of the LLaMA-350M model trained in a long-context window setting.*

### Benchmark 3: Pretraining LLaMA-7B model within 16GB memory

The command of training LLaMA-7B model on single GPU as provided within `scripts/single_gpu`. With 1 batch size, the following scripts can pre-train a LLaMA-7B model within 11GB memory (tested on a single A100 GPU)

### Benchmark 4: Memory-efficient full-parameter LLM finetuning

Now we support APOLLO in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We have added a test in the `examples/extras/apollo` directory. 

We conducted a comparative evaluation with **GaLore** by fine-tuning models and testing on the **MMLU task**.

#### GaLore Performance using `examples/extras/galore`
```
Average: 64.96
           STEM: 55.43
Social Sciences: 75.66
     Humanities: 59.72
          Other: 71.25
```


#### APOLLO Performance (Scaling Factor = 32) using `examples/extras/apollo`
With a scaling factor derived from the ratio of LLaMA-8B dimension (4096) to rank (128):
```
Average: 65.03
           STEM: 55.47
Social Sciences: 76.15
     Humanities: 59.60
          Other: 71.28
```
---


## License

The majority of APOLLO is licensed under CC-BY-NC, however portions of the project are available under separate license terms: GaLore is licensed under the Apache 2.0 license.


## Acknowledgements

* The above code is based on the codebase of [GaLore](https://github.com/jiaweizzhao/GaLore) and [Q-GaLore](https://github.com/VITA-Group/Q-GaLore).
* We'd like to express our gratitude to [Fira](https://github.com/xichen-fy/Fira) for their invetion of norm-growth-limiter.
* We'd like to express our gratitude to [@murrellb](https://github.com/murrellb) for the pull request to FluxML! 
