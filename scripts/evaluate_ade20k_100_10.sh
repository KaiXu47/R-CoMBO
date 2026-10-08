#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${1:-0}"
RUN_DIR="${2:-output/ps/100-10}"
TASK="${3:-6}"
PY="${PY:-python}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export EXPORT_PANOPTIC_PER_CLASS=1
cd "${ROOT}"

if (( TASK < 1 || TASK > 6 )); then
  echo "TASK must be between 1 and 6." >&2
  exit 1
fi
"${PY}" train_continual.py --num-gpus "${NUM_GPUS}" --eval-only \
  --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
  CONT.TASK "${TASK}" OUTPUT_DIR "${RUN_DIR}/step${TASK}"
