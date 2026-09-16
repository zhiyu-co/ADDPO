#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/ad2po}"
METHOD="${METHOD:-joint}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Training data not found: ${DATA_FILE}" >&2
  exit 1
fi
if [[ "${METHOD}" != "grpo" && "${METHOD}" != "trajectory" && \
      "${METHOD}" != "token" && "${METHOD}" != "joint" ]]; then
  echo "METHOD must be one of: grpo, trajectory, token, joint" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${ROOT}/src/train.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_file "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG:-${ROOT}/conf/zero3.json}" \
  --method "${METHOD}" \
  --think_max_len "${THINK_MAX_LEN:-128}" \
  --beta "${BETA:-0.01}" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --max_steps "${MAX_STEPS:-338}" \
  --save_steps "${SAVE_STEPS:-338}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --seed "${SEED:-42}" \
  --freeze_audio_encoder "${FREEZE_AUDIO_ENCODER:-true}" \
  --report_to "${REPORT_TO:-swanlab}" \
  --run_name "${RUN_NAME:-qwen2.5-omni-7b-${METHOD}-seed${SEED:-42}}"
