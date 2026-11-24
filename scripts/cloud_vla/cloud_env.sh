#!/usr/bin/env bash
# Source this file in each cloud terminal before preparing assets or training.
task_bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
task_venv_dir=${ROBOTARM_VENV_DIR:-${task_bundle_root}/.venv-cloud}
if [[ ! -f ${task_venv_dir}/bin/activate ]]; then
  echo "Missing ${task_venv_dir}; run scripts/cloud_vla/setup_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
source "${task_venv_dir}/bin/activate"
export HF_HOME="${HF_HOME:-${task_bundle_root}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-robotarm-vla}"
export OMP_NUM_THREADS="${ROBOTARM_OMP_NUM_THREADS:-4}"
if [[ ${ROBOTARM_OFFLINE:-0} == 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
cd -- "${task_bundle_root}"
