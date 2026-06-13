#!/bin/bash
# Stage 2 — FineWeb-Edu 1B, best hyperparameters per optimizer
# 4 archs: transformer / gla / deltanet / gated_deltanet
# 8 GPUs, 1 node | 30720 steps | seq_len=32768 | grad_acc=2
#
# Usage: bash scripts/run_1b.sh <arch> <optimizer>
# Example: bash scripts/run_1b.sh gla muon

ARCH=${1:-transformer}
OPT=${2:-adamw}
NGPU=${NGPU:-8}

# ── paths (edit before running) ──────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_PATH="/PATH/TO/fineweb-edu/sample/100BT"
TOKENIZER_PATH="/PATH/TO/tokenizer"
OUTPUT_BASE="./exp/1b_fwe"
VAL_DATA_DIR="/PATH/TO/wiki_val"
# ─────────────────────────────────────────────────────────────────────────────

export HF_ENDPOINT=https://hf-mirror.com

case $ARCH in
  transformer)    MODEL_CONFIG="configs/transformer_1B.json" ;;
  gla)            MODEL_CONFIG="configs/gla_1B.json" ;;
  deltanet)       MODEL_CONFIG="configs/delta_net_1B.json" ;;
  gated_deltanet) MODEL_CONFIG="configs/gated_deltanet_1B.json" ;;
  *)
    echo "Unknown arch: $ARCH"
    echo "Available: transformer gla deltanet gated_deltanet"
    exit 1 ;;
esac

case $OPT in
  adamw)
    OPT_NAME=AdamW LR=1e-3 B1=0.9  B2=0.99 EPS=1e-15 ;;
  adamp)
    OPT_NAME=AdamP LR=1e-3 B1=0.9  B2=0.98 EPS=1e-15 ;;
  adan)
    OPT_NAME=Adan  LR=3e-3 B1=0.9  B2=0.92 EPS=1e-8
    export OPT_BETA3=0.99 ;;
  lion)
    OPT_NAME=Lion  LR=3e-4 B1=0.9  B2=0.99 EPS="" ;;
  mars_adamw)
    OPT_NAME=MARS  LR=3e-3 B1=0.95 B2=0.99 EPS=1e-8
    export MARS_TYPE=mars-adamw ;;
  mars_lion)
    OPT_NAME=MARS  LR=2e-4 B1=0.9  B2=0.98 EPS=1e-8
    export MARS_TYPE=mars-lion ;;
  mars_shampoo)
    OPT_NAME=MARS  LR=1e-2 B1=0.95 B2=0.99 EPS=1e-8
    export MARS_TYPE=mars-shampoo ;;
  muon)
    OPT_NAME=Muon  LR=5e-3 B1=0.9  B2=0.95 EPS=1e-15 ;;
  rmnp)
    OPT_NAME=RMNP  LR=5e-3 B1=0.9  B2=0.99 EPS=1e-15
    export ADAM_LR=1e-3 ;;
  soap)
    OPT_NAME=SOAP  LR=3e-3 B1=0.9  B2=0.95 EPS=1e-15 ;;
  apollo)
    OPT_NAME=APOLLO_AdamW LR=3e-3 B1=0.9 B2=0.99 EPS=1e-12
    export APOLLO_RANK=512 APOLLO_SCALE=2 APOLLO_UPDATE_PROJ_GAP=200
    export APOLLO_SCALE_TYPE=channel APOLLO_PROJ=random APOLLO_PROJ_TYPE=std ;;
  conda)
    OPT_NAME=Conda LR=5e-4 B1=0.9  B2=0.99 EPS=1e-8
    export CONDA_RANK=256 CONDA_SCALE=1.0 CONDA_UPDATE_PROJ_GAP=200 CONDA_PROJ_TYPE=std ;;
  *)
    echo "Unknown optimizer: $OPT"
    echo "Available: adamw adamp adan lion mars_adamw mars_lion mars_shampoo muon rmnp soap apollo conda"
    exit 1 ;;
esac

EXPERIMENT_NAME="${ARCH}_1b_fwe_${OPT}"
DUMP_FOLDER="${OUTPUT_BASE}/${EXPERIMENT_NAME}/exp_data"
LOG_FILE="${OUTPUT_BASE}/logs/${EXPERIMENT_NAME}.log"

mkdir -p "${OUTPUT_BASE}/${EXPERIMENT_NAME}" "$(dirname "${LOG_FILE}")"
cd "${PROJECT_ROOT}"

TRAIN_PARAMS=(
  --job.config_file flame/models/fla.toml
  --job.dump_folder "${DUMP_FOLDER}"
  --model.config "${MODEL_CONFIG}"
  --model.tokenizer_path "${TOKENIZER_PATH}"
  --optimizer.name "${OPT_NAME}"
  --optimizer.lr "${LR}"
  --optimizer.beta1 "${B1}"
  --optimizer.beta2 "${B2}"
  --lr_scheduler.decay_type cosine
  --lr_scheduler.warmup_steps 1024
  --lr_scheduler.lr_min 0.1
  --training.batch_size 1
  --training.seq_len 32768
  --training.context_len 4096
  --training.gradient_accumulation_steps 2
  --training.steps 30720
  --training.varlen
  --training.max_norm 1.0
  --training.skip_nan_inf
  --training.dataset "${DATASET_PATH}"
  --training.dataset_name default
  --training.dataset_split train
  --training.streaming
  --training.num_workers 4
  --training.prefetch_factor 2
  --training.seed 42
  --training.data_parallel_replicate_degree "${NGPU}"
  --training.data_parallel_shard_degree 1
  --training.val_times 30
  --training.val_data_dir "${VAL_DATA_DIR}"
  --checkpoint.interval 20000
  --checkpoint.load_step -1
  --checkpoint.keep_latest_k 2
  --metrics.log_freq 1
)

[[ -n "${EPS}" ]] && TRAIN_PARAMS+=(--optimizer.eps "${EPS}")

bash train.sh "${TRAIN_PARAMS[@]}" 2>&1 | tee "${LOG_FILE}"
