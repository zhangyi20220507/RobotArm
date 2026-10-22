#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/cloud_env.sh"
robotarm modeling train-pi0-lora -- \
  --dataset-root data/lerobot/piper_pick_cup_dual_camera_lf_180eps \
  --base-checkpoint artifacts/pretrained/pi0_base \
  --output-dir artifacts/checkpoints/pi0_lora_lf180_v1 \
  --batch-size 1 --steps 20000 --rank 16 --alpha 32 --lr 0.0001 \
  --chunk-size 50 --n-action-steps 5 --eval-split 0.2 --seed 1000 \
  --save-freq 1000 --eval-steps 1000 --max-eval-samples 256 "$@"
