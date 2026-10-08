#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${1:-0,1}"
RUN_DIR="${2:-output/ps/100-10}"
PY="${PY:-python}"
SEED="${SEED:-26388850}"
START_TASK="${START_TASK:-1}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_GPUS="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export EXPORT_PANOPTIC_PER_CLASS=1
cd "${ROOT}"

if (( START_TASK < 1 || START_TASK > 6 )); then
  echo "START_TASK must be between 1 and 6." >&2
  exit 1
fi

for task in $(seq "${START_TASK}" 6); do
  WEIGHTS=()
  if (( task == 1 )); then
    BASE_LR=0.0001
    MAX_ITER=160000
  else
    BASE_LR=0.00005
    MAX_ITER=10000
    PREVIOUS="${RUN_DIR}/step$((task - 1))/model_final.pth"
    if [[ ! -f "${PREVIOUS}" ]]; then
      echo "Missing previous task checkpoint: ${PREVIOUS}" >&2
      exit 1
    fi
    WEIGHTS=(CONT.WEIGHTS "${PREVIOUS}" CONT.OLD_WEIGHTS "${PREVIOUS}")
  fi
  "${PY}" train_continual.py --num-gpus "${NUM_GPUS}" --dist-url auto \
    --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
    CONT.TASK "${task}" SEED "${SEED}" \
    SOLVER.BASE_LR "${BASE_LR}" SOLVER.MAX_ITER "${MAX_ITER}" \
    "${WEIGHTS[@]}" OUTPUT_DIR "${RUN_DIR}/step${task}"
done
