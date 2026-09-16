#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/MMAU/mmau-test-mini.local.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/mmau}"
THINK="${THINK:-true}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: MODEL_PATH=/path/to/model bash evaluate_mmau.sh" >&2
  exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Prepared MMAU metadata not found: ${DATA_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/src/test.py" \
  --model_path "${MODEL_PATH}" \
  --data_file "${DATA_FILE}" \
  --out_file "${OUTPUT_DIR}/predictions.json" \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
  --think "${THINK}" \
  --think_max_len "${THINK_MAX_LEN:-128}"

python "${MMAU_ROOT:-${ROOT}/data/MMAU}/evaluation.py" \
  --input "${OUTPUT_DIR}/predictions.json" \
  | tee "${OUTPUT_DIR}/metrics.txt"
