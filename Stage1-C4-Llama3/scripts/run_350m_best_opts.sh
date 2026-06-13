#!/bin/bash
# Stage 1 — LLaMA-3 350M on C4, best hyperparameters per optimizer
# 4 GPUs, 1 node | 60k steps | seq_len=256 | total_batch_size=512
#
# Usage: bash scripts/run_350m_best_opts.sh <optimizer>
# Example: bash scripts/run_350m_best_opts.sh muon

OPT=${1:-adamw}
GPUS=${GPUS:-4}
COMMON="--model_config configs/llama_350m.json \
        --batch_size 128 --total_batch_size 512 \
        --num_training_steps 60000 --warmup_steps 6000 \
        --weight_decay 0.0 --dtype bfloat16 \
        --eval_every 1000 --scheduler cosine --min_lr_ratio 0.1 \
        --project stage1_350m --unset_wandb"

case $OPT in
  adamw)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adamw --lr 1e-3 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  adabelief)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adabelief --lr 1e-3 --beta1 0.9 --beta2 0.999 --eps 1e-16 ;;
  adafactor)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adafactor --lr 1e-3 --beta1 0.9 --beta2 0.999 ;;
  adam8bit)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adam8bit --lr 5e-4 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  adam_mini)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adam_mini --lr 5e-4 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  adamp)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adamp --lr 1e-3 --beta1 0.9 --beta2 0.98 --eps 1e-8 ;;
  adan)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer adan --lr 3e-3 --beta1 0.9 --beta2 0.92 --beta3 0.99 --eps 1e-8 ;;
  came)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer came --lr 5e-4 --beta1 0.9 --beta2 0.999 --eps 1e-6 ;;
  conda)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer conda --lr 1e-2 --beta1 0.9 --beta2 0.99 --eps 1e-8 \
      --rank 256 --update_proj_gap 2000 --apollo_scale 0.25 ;;
  galore)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer galore_adamw --lr 1e-2 --beta1 0.9 --beta2 0.98 --eps 1e-6 \
      --rank 256 --update_proj_gap 500 --galore_scale 0.25 ;;
  lamb)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer lamb --lr 1e-3 --beta1 0.9 --beta2 0.99 --eps 1e-6 ;;
  lion)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer lion --lr 2e-4 --beta1 0.9 --beta2 0.98 ;;
  mars_adamw)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer mars_adamw --lr 5e-3 --beta1 0.95 --beta2 0.99 --eps 1e-8 ;;
  mars_lion)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer mars_lion --lr 2e-4 --beta1 0.9 --beta2 0.98 --eps 1e-8 ;;
  mars_shampoo)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer mars_shampoo --lr 2e-2 --beta1 0.95 --beta2 0.99 --eps 1e-8 ;;
  muon)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer muon --lr 6e-3 --beta1 0.9 --beta2 0.95 --eps 1e-8 ;;
  nadam)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer nadam --lr 1e-3 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  prodigy)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer prodigy --lr 2.0 --beta1 0.9 --beta2 0.95 --eps 1e-8 ;;
  radam)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer radam --lr 1e-3 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  rmnp)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer rmnp --lr 1e-2 --beta1 0.9 --beta2 0.95 --eps 1e-8 ;;
  shampoo)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer shampoo --lr 2e-2 --beta1 0.9 --beta2 0.999 --eps 1e-8 ;;
  soap)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer soap --lr 1e-3 --beta1 0.9 --beta2 0.95 --eps 1e-8 ;;
  sophia)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer sophia --lr 2e-4 --beta1 0.9 --beta2 0.99 --eps 1e-8 ;;
  apollo)
    torchrun --standalone --nproc_per_node $GPUS torchrun_main.py $COMMON \
      --optimizer apollo_adamw --lr 1e-2 --beta1 0.9 --beta2 0.99 --eps 1e-6 \
      --rank 256 --update_proj_gap 200 --scale_type channel --proj random \
      --apollo_scale 1.0 ;;
  *)
    echo "Unknown optimizer: $OPT"
    echo "Available: adamw adabelief adafactor adam8bit adam_mini adamp adan came conda galore lamb lion mars_adamw mars_lion mars_shampoo muon nadam prodigy radam rmnp shampoo soap sophia apollo"
    exit 1 ;;
esac
