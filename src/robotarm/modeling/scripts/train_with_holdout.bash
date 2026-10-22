#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_ROOT OUTPUT_DIR [LEROBOT_TRAIN_ARGS...]" >&2
  exit 2
fi

dataset_root=$(realpath -e "$1")
output_dir=$2
shift 2

# Holdout settings come from src/robotarm/config/modeling.yaml
# (train-with-holdout section). The default eval_split 0.1935483870967742
# (6/31) holds out exactly the last six complete episodes; LeRobot performs
# the split by episode before constructing either dataloader.
holdout_cfg=$(python -c '
from robotarm.config import defaults
cfg = defaults("modeling", "train-with-holdout")
print(cfg["policy-type"], cfg["eval-split"], cfg["eval-steps"], cfg["max-eval-samples"])
')
read -r holdout_policy holdout_split holdout_eval_steps holdout_max_samples <<< "${holdout_cfg}"

exec "$(dirname "$0")/train.bash" \
  "$dataset_root" "$output_dir" "${holdout_policy}" \
  --dataset.eval_split="${holdout_split}" \
  --eval_steps="${holdout_eval_steps}" \
  --max_eval_samples="${holdout_max_samples}" \
  "$@"
