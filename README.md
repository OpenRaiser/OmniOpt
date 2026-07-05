# OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers

A systematic two-stage benchmark comparing optimizers for LLM pre-training at scale.

**Stage 1 (Broad Screening)** sweeps 24+ optimizers on C4 under the LLaMA-3 architecture at four scales — 60M, 130M, 350M, and 1B — using final C4 validation perplexity.

**Stage 2 (High-Quality Generalization)** transfers the stronger Stage-1 optimizers to FineWeb-Edu with 32k sequences, at 340M and 1B, across four architectures: Transformer++, GLA, DeltaNet, and Gated DeltaNet.

A strict controlled-variable protocol is used: only optimizer hyperparameters (`lr`, `betas`, `eps`, method-specific knobs) are tuned per optimizer; all architectural, data, and schedule settings are held fixed.

---

## Repository Structure

```
OmniOpt/
├── Stage1-C4-Llama3/
│   ├── torchrun_main.py     # main training entry point
│   ├── opt/                 # 24+ optimizer implementations
│   ├── utils/               # dataloader, modeling, training utilities
│   ├── configs/             # LLaMA-3 model configs (60M–1B)
│   └── scripts/             # best-hyperparameter launch scripts
│       ├── run_350m_best_opts.sh
│       └── run_1b_best_opts.sh
│
└── Stage2-FWE/
    ├── trainer/flame/       # FLA-based training framework
    │   └── scripts/
    │       ├── run_340m.sh  # best-hyperparameter launch script (340M, 4 archs)
    │       └── run_1b.sh    # best-hyperparameter launch script (1B, 4 archs)
    ├── opentome/            # optimizer integrations and model implementations
    └── evaluations/         # downstream evaluation framework
```

---

## Optimizers Covered

**Stage 1**: AdamW, AdaBelief, Adafactor, Adam8bit, Adam-mini, AdamP, Adan, CAME, Conda, GaLore, LAMB, Lion, MARS-AdamW, MARS-Lion, MARS-Shampoo, Muon, NAdam, Prodigy, RAdam, RMNP, Shampoo, SOAP, Sophia, APOLLO.

**Stage 2** (carried forward): AdamW, AdamP, Adan, Lion, MARS-AdamW, MARS-Lion, MARS-Shampoo, Muon, RMNP, SOAP, APOLLO, Conda.

---

## Environment Setup

### Stage 1 (C4 / LLaMA-3)

```bash
cd Stage1-C4-Llama3
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.1, Transformers, Datasets, bitsandbytes (optional, for 8-bit optimizers).

To use a HuggingFace mirror (e.g. in mainland China):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Stage 2 (FineWeb-Edu / multi-arch)

```bash
cd Stage2-FWE
conda env create -f fla_environment.yml
conda activate fla
pip install -e .
```

Requires Flash Linear Attention (FLA) for GLA / DeltaNet / Gated DeltaNet architectures.

---

## Running Experiments

### Stage 1

```bash
cd Stage1-C4-Llama3

# 350M (60k steps, 4 GPUs)
bash scripts/run_350m_best_opts.sh muon
bash scripts/run_350m_best_opts.sh apollo

# 1B (100k steps, 8 GPUs)
bash scripts/run_1b_best_opts.sh muon
bash scripts/run_1b_best_opts.sh apollo
```

Or launch manually:

```bash
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --optimizer muon --lr 6e-3 --beta1 0.9 --beta2 0.95 --eps 1e-8 \
    --batch_size 128 --total_batch_size 512 \
    --num_training_steps 60000 --warmup_steps 6000 \
    --weight_decay 0.0 --dtype bfloat16
```

### Stage 2

Edit the three path variables at the top of the scripts (`DATASET_PATH`, `TOKENIZER_PATH`, `VAL_DATA_DIR`), then:

```bash
cd Stage2-FWE/trainer/flame

# 340M (30720 steps, 8 GPUs)
bash scripts/run_340m.sh gla muon
bash scripts/run_340m.sh transformer apollo

# 1B (30720 steps, 8 GPUs)
bash scripts/run_1b.sh gla muon
bash scripts/run_1b.sh transformer apollo
```

Available architectures: `transformer`, `gla`, `deltanet`, `gated_deltanet`

Available optimizers: `adamw`, `adamp`, `adan`, `lion`, `mars_adamw`, `mars_lion`, `mars_shampoo`, `muon`, `rmnp`, `soap`, `apollo`, `conda`

---

## Training Protocol

| Stage | Dataset | Architecture | Scales | Seq Len | Steps |
|---|---|---|---|---|---|
| 1 | C4 | LLaMA-3 | 60M, 130M, 350M, 1B | 256 | 10k / 20k / 60k / 100k |
| 2 | FineWeb-Edu | Transformer++, GLA, DeltaNet, Gated DeltaNet | 340M, 1B | 32k | ~30k |

Cosine LR schedule, linear warmup (10% of steps), weight decay 0.0, gradient clipping 1.0.

---

## Acknowledgements

- **Stage 1** training framework is built on [APOLLO](https://github.com/zhuhanqing/APOLLO), which is based on [GaLore](https://github.com/jiaweizzhao/GaLore) and [Q-GaLore](https://github.com/VITA-Group/Q-GaLore).
- **Stage 2** training framework is built on [OpenToMe](https://github.com/Westlake-AI/OpenToMe), which integrates [flame](https://github.com/fla-org/flame) from the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) project.

---

## Citation

```bibtex
@article{li2025omniopt,
  title  = {{OmniOpt: Taxonomy, Geometry, and Benchmarking of Modern Optimizers}},
  author = {Siyuan Li and Jiabao Pan and Yumou Liu and Zhuoli Ouyang and Xin Jin and Xinglong Xu and Jingxuan Wei and Shengye Pang and Jintao Chen and Xuanhe Zhou and Conghui He and Cheng Tan},
  year   = {2025},
}
```
