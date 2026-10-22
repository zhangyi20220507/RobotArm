#!/usr/bin/env bash
# Source this file after unpacking, including in every new cloud terminal.
TASK_BUNDLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export HF_HOME="${TASK_BUNDLE_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
if [[ -f "${TASK_BUNDLE_ROOT}/.venv-pi0/bin/activate" ]]; then
  source "${TASK_BUNDLE_ROOT}/.venv-pi0/bin/activate"
fi
cd -- "${TASK_BUNDLE_ROOT}"
