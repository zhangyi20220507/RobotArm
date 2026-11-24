#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

default_checkpoint="artifacts/checkpoints/20261115_act_flvv_267eps_layer4_chunk50_bs16_160k_v1/checkpoints/last/pretrained_model"
checkpoint=${ACT_CHECKPOINT:-${default_checkpoint}}
report=${ACT_REPORT:-}
duration=${ACT_DURATION:-150}
interface=${ACT_INTERFACE:-socketcan}
channel=${ACT_CHANNEL:-can_follower1}
firmware=${ACT_FIRMWARE:-v189}
global_camera=${ACT_GLOBAL_CAMERA:-347522071736}
wrist_camera=${ACT_WRIST_CAMERA:-260322272554}
device=${ACT_DEVICE:-cuda}
robotarm_bin=${ROBOTARM_BIN:-robotarm}

mode=shadow
dry_run=0
extra_args=()

usage() {
  cat <<'EOF'
Run a trained ACT checkpoint through RobotArm's guarded online-policy loop.

Usage:
  scripts/run_act_policy.sh [--shadow] [--execute] [--dry-run]
                            [--checkpoint PATH] [--report PATH]
                            [--duration SECONDS] [-- EXTRA_ONLINE_POLICY_ARGS...]

Modes:
  --shadow    Read cameras and arm feedback, run inference, but send no motion
              commands. This is the default.
  --execute   Enable real arm commands. The arm first homes and remains paused;
              press X in the preview window to start one policy trial.
  --dry-run   Validate local files and print the final command without running it.

Environment overrides:
  ACT_CHECKPOINT, ACT_REPORT, ACT_DURATION, ACT_INTERFACE, ACT_CHANNEL,
  ACT_FIRMWARE, ACT_GLOBAL_CAMERA, ACT_WRIST_CAMERA, ACT_DEVICE, ROBOTARM_BIN

Examples:
  scripts/run_act_policy.sh --shadow --duration 60
  scripts/run_act_policy.sh --execute
  ACT_CHECKPOINT=artifacts/checkpoints/RUN/checkpoints/STEP/pretrained_model \
    scripts/run_act_policy.sh --shadow
EOF
}

require_value() {
  if [[ $# -lt 2 || -z ${2} ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shadow)
      mode=shadow
      ;;
    --execute)
      mode=execute
      ;;
    --dry-run)
      dry_run=1
      ;;
    --checkpoint)
      require_value "$@"
      checkpoint=$2
      shift
      ;;
    --report)
      require_value "$@"
      report=$2
      shift
      ;;
    --duration)
      require_value "$@"
      duration=$2
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      ;;
  esac
  shift
done

if [[ ${checkpoint} != /* ]]; then
  checkpoint="${workspace}/${checkpoint}"
fi
if [[ ! -d ${checkpoint} ]]; then
  echo "ACT checkpoint directory not found: ${checkpoint}" >&2
  exit 1
fi
checkpoint=$(realpath -e -- "${checkpoint}")

for required in config.json train_config.json model.safetensors \
  policy_preprocessor.json policy_postprocessor.json; do
  if [[ ! -s ${checkpoint}/${required} ]]; then
    echo "ACT checkpoint is incomplete; missing ${checkpoint}/${required}" >&2
    exit 1
  fi
done
if ! grep -Eq '"type"[[:space:]]*:[[:space:]]*"act"' "${checkpoint}/config.json"; then
  echo "Checkpoint config is not an ACT policy: ${checkpoint}/config.json" >&2
  exit 1
fi

if [[ -z ${report} ]]; then
  step_name=$(basename -- "$(dirname -- "${checkpoint}")")
  run_name=$(basename -- "$(dirname -- "$(dirname -- "$(dirname -- "${checkpoint}")")")")
  timestamp=$(date +%Y%m%d_%H%M%S)
  report="artifacts/logs/shadow/${run_name}_${step_name}_${mode}_${timestamp}.jsonl"
fi
if [[ ${report} != /* ]]; then
  report="${workspace}/${report}"
fi
mkdir -p -- "$(dirname -- "${report}")"

command=(
  "${robotarm_bin}" modeling online-policy --
  --checkpoint "${checkpoint}"
  --interface "${interface}"
  --channel "${channel}"
  --firmware "${firmware}"
  --camera-serial "${global_camera}"
  --wrist-camera-serial "${wrist_camera}"
  --device "${device}"
  --rate 20
  --camera-fps 20
  --n-action-steps 1
  --temporal-ensemble-coeff 0.01
  --warn-inference-ms 60
  --max-inference-ms 500
  --warn-act-raw-joint-gap-deg 12
  --max-act-raw-joint-gap-deg 15
  --max-act-raw-gap-consecutive-steps 3
  --max-act-subgoal-no-progress-steps 10
  --duration "${duration}"
  --report "${report}"
)
if [[ ${mode} == execute ]]; then
  command+=(--execute)
fi
command+=("${extra_args[@]}")

printf 'Mode:       %s\n' "${mode}"
printf 'Checkpoint: %s\n' "${checkpoint}"
printf 'Report:     %s\n' "${report}"
printf 'Command:    '
printf '%q ' "${command[@]}"
printf '\n'

if [[ ${dry_run} == 1 ]]; then
  exit 0
fi
if ! command -v -- "${robotarm_bin}" >/dev/null 2>&1; then
  echo "RobotArm executable not found: ${robotarm_bin}" >&2
  echo "Activate the robotarm environment or set ROBOTARM_BIN to its executable." >&2
  exit 1
fi
if [[ ${interface} == socketcan && ! -d /sys/class/net/${channel} ]]; then
  echo "SocketCAN interface does not exist: ${channel}" >&2
  echo "Run the Piper/CAN setup first, then verify it with: ip -details link show ${channel}" >&2
  exit 1
fi

if [[ ${mode} == execute ]]; then
  printf '%s\n' \
    'REAL ARM EXECUTION ENABLED.' \
    'Keep the emergency stop reachable and clear the workspace.' \
    'The runner homes first and remains paused; press X only when the scene is ready.'
else
  printf '%s\n' 'Shadow mode: inference runs, but policy motion commands are not sent.'
fi

cd -- "${workspace}"
exec "${command[@]}"
