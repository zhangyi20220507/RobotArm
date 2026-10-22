#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/cloud_env.sh"
robotarm modeling train-pi0-lora -- \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --output-dir artifacts/checkpoints/pi0_lora_lf180_smoke \
  --smoke "$@"
