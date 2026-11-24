#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "${bundle_root}/scripts/cloud_vla/cloud_env.sh"
job_file=${ROBOTARM_JOB_FILE:-${bundle_root}/job.env}
if [[ ! -f ${job_file} ]]; then
  echo "Missing ${job_file}; copy job.env.example to job.env and edit it." >&2
  exit 1
fi
set -a
source "${job_file}"
set +a

required=(POLICY DATASET_ROOT BASE_CHECKPOINT OUTPUT_DIR BATCH_SIZE STEPS CHUNK_SIZE N_ACTION_STEPS)
for name in "${required[@]}"; do
  if [[ -z ${!name:-} ]]; then
    echo "Missing ${name} in ${job_file}" >&2
    exit 1
  fi
done

common_args=(
  --dataset-root "${DATASET_ROOT}"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --chunk-size "${CHUNK_SIZE}"
  --n-action-steps "${N_ACTION_STEPS}"
  --eval-split "${EVAL_SPLIT:-0.1}"
  --save-freq "${SAVE_FREQ:-2000}"
  --eval-steps "${EVAL_STEPS:-2000}"
  --max-eval-samples "${MAX_EVAL_SAMPLES:-256}"
  --rank "${LORA_RANK:-16}"
  --alpha "${LORA_ALPHA:-32}"
  --lr "${LEARNING_RATE:-0.0001}"
  --num-workers "${NUM_WORKERS:-4}"
  --seed "${SEED:-1000}"
)
if [[ -n ${QUALITY_REVIEWS:-} ]]; then
  common_args+=(--quality-reviews "${QUALITY_REVIEWS}")
fi

case "${POLICY}" in
  pi0)
    exec robotarm modeling train-pi0-lora -- "${common_args[@]}" "$@"
    ;;
  smolvla)
    exec robotarm modeling train-smolvla -- "${common_args[@]}" \
      --training-mode "${TRAINING_MODE:-expert}" "$@"
    ;;
  *)
    echo "POLICY must be pi0 or smolvla, got: ${POLICY}" >&2
    exit 2
    ;;
esac
