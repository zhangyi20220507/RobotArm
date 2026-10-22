#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v robotarm >/dev/null 2>&1; then
  echo "robotarm command not found; activate the robotarm environment first." >&2
  exit 1
fi

# Intentionally replaces the previous 100-episode simulation dataset.
# Extra CLI arguments are appended so parameters such as --seed can be
# overridden without editing this file.
exec robotarm simulation collect-lerobot -- \
  --episodes 100 \
  --max-trials 400 \
  --action-frame absolute \
  --max-joint-delta-rad 0.4 \
  --max-gripper-delta-m 0.01 \
  --max-episode-frames 650 \
  --width 320 --height 240 \
  --output data/simulation/lerobot/piper_pick_cup_joint_absolute_100 \
  --repo-id local/sim-piper-pick-cup-joint-absolute-100 \
  --overwrite \
  "$@"
