#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/cloud_env.sh"
robotarm modeling check-pi0-lora -- \
  --checkpoint artifacts/checkpoints/pi0_lora_lf180_smoke/checkpoints/000020/pretrained_model \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --index 0 --output artifacts/reports/pi0_lora_smoke_reload.json "$@"
