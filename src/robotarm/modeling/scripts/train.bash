#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_ROOT OUTPUT_DIR [POLICY_TYPE] [LEROBOT_TRAIN_ARGS...]" >&2
  exit 2
fi

dataset_root=$(realpath -e "$1")
output_dir=$2
# Default policy type comes from src/robotarm/config/modeling.yaml (train.policy-type).
default_policy_type=$(python -c '
from robotarm.config import defaults
print(defaults("modeling", "train")["policy-type"])
')
policy_type=${3:-${default_policy_type}}
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

# Dataset Browser reviews are stored separately from the dataset.  Opt in to
# using them for training; keep and unreviewed episodes are selected, while
# review and reject episodes are excluded.
quality_filter=false
quality_reviews=""
quality_shuffle_seed=""
custom_episodes=false
lerobot_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quality-filter)
      quality_filter=true
      shift
      ;;
    --quality-shuffle-seed)
      if [[ $# -lt 2 ]]; then
        echo "--quality-shuffle-seed requires an integer seed." >&2
        exit 2
      fi
      quality_filter=true
      quality_shuffle_seed=$2
      shift 2
      ;;
    --quality-shuffle-seed=*)
      quality_filter=true
      quality_shuffle_seed=${1#*=}
      shift
      ;;
    --quality-reviews)
      if [[ $# -lt 2 ]]; then
        echo "--quality-reviews requires a JSON path." >&2
        exit 2
      fi
      quality_filter=true
      quality_reviews=$2
      shift 2
      ;;
    --quality-reviews=*)
      quality_filter=true
      quality_reviews=${1#*=}
      shift
      ;;
    --dataset.episodes|--dataset.episodes=*)
      custom_episodes=true
      lerobot_args+=("$1")
      shift
      ;;
    *)
      lerobot_args+=("$1")
      shift
      ;;
  esac
done

if [[ ${quality_filter} == true ]]; then
  if [[ ${custom_episodes} == true ]]; then
    echo "Do not combine --quality-filter/--quality-reviews with --dataset.episodes." >&2
    exit 2
  fi
  quality_command=(python -m robotarm.modeling.quality_filter --dataset-root "${dataset_root}")
  if [[ -n ${quality_reviews} ]]; then
    quality_command+=(--reviews "${quality_reviews}")
  fi
  if [[ -n ${quality_shuffle_seed} ]]; then
    quality_command+=(--shuffle-seed "${quality_shuffle_seed}")
  fi
  selected_episodes=$("${quality_command[@]}")
  lerobot_args+=("--dataset.episodes=${selected_episodes}")
fi

# Keep Hugging Face locks and downloaded artifacts inside the shared workspace.
# The host-level ~/.cache may be read-only in managed training environments.
workspace_cache=$(realpath -m "$(dirname "$0")/../../../../.cache/huggingface")
mkdir -p "${workspace_cache}/datasets"
export HF_HOME="${HF_HOME:-${workspace_cache}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

info_path="${dataset_root}/meta/info.json"
if [[ ! -f ${info_path} ]]; then
  echo "Not a LeRobot dataset (missing ${info_path})." >&2
  exit 1
fi
if [[ ${policy_type} == act ]]; then
  python -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
features = set(json.loads((root / "meta/info.json").read_text(encoding="utf-8"))["features"])
observations = {key for key in features if key.startswith("observation.")}
images = {key for key in observations if key.startswith("observation.images.")}
expected = {"observation.state"} | images
if not images:
    raise SystemExit("ACT dataset must contain at least one observation.images.* feature")
if observations != expected:
    raise SystemExit(
        "ACT dataset has unsafe/unexpected observation features: "
        + ", ".join(sorted(observations - expected))
    )
if "action" not in features:
    raise SystemExit("ACT dataset is missing action")
print("ACT schema OK: " + " + ".join(sorted(expected)) + " -> action")
' "${dataset_root}"
fi
repo_id=$(python -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
provenance_path = root / "meta/piper_conversion.json"
provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
print(provenance.get("repo_id") or info.get("repo_id") or f"local/{root.name}")
' "${dataset_root}")
# LeRobot refuses to start a new run when the output directory already exists.
# Resolve the intended path without creating it; only ensure its parent exists.
output_dir=$(realpath -m "${output_dir}")
mkdir -p "$(dirname "${output_dir}")"
job_name="${policy_type}_$(basename "${dataset_root}")"
policy_repo_id="local/${job_name}"

echo "Dataset: ${dataset_root}"
echo "Repo id: ${repo_id}"
echo "Policy: ${policy_type}"
echo "Policy repo id: ${policy_repo_id}"
echo "Output: ${output_dir}"

# Keep wandb on so loss curves are persisted. Without an API key, fall back to
# offline mode: runs are written to ./wandb/ locally and can be synced later
# with `wandb sync <run_dir>` once logged in.
# NOTE: lerobot's WandBLogger hardcodes mode=... else "online" and ignores the
# WANDB_MODE env var, so the mode MUST be passed as a CLI flag here.
if [[ -z ${WANDB_API_KEY:-} && -z ${WANDB_MODE:-} ]]; then
  export WANDB_MODE=offline
fi
if [[ -z ${WANDB_PROJECT:-} ]]; then
  export WANDB_PROJECT=robotarm
fi
wandb_mode=${WANDB_MODE:-online}
echo "Wandb: enabled (mode=${wandb_mode})"

exec lerobot-train \
  --dataset.repo_id="${repo_id}" \
  --dataset.root="${dataset_root}" \
  --policy.type="${policy_type}" \
  --policy.repo_id="${policy_repo_id}" \
  --policy.push_to_hub=false \
  --output_dir="${output_dir}" \
  --job_name="${job_name}" \
  --wandb.enable=true \
  --wandb.mode="${wandb_mode}" \
  "${lerobot_args[@]}"
