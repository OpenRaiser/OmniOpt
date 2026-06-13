#!/usr/bin/bash
# Stage 2 — FineWeb-Edu training via flame
# See trainer/flame/scripts/ for per-optimizer launch scripts.
# Example: 340M GLA with AdamW on 1 node / 8 GPUs

export HF_ENDPOINT=https://hf-mirror.com

NNODE=1 NGPU=8 LOG_RANK=0 bash trainer/flame/train.sh \
  --job.config_file trainer/flame/flame/models/fla.toml \
  --job.dump_folder RESULTS/PATH \
  --model.config trainer/flame/configs/gla_340M.json \
  --model.tokenizer_path TOKENIZER/PATH \
  --optimizer.name AdamW \
  --optimizer.eps 1e-8 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 32768 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 30000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset DATASET/PATH \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1
