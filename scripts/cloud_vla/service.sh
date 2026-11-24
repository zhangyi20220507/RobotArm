#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "${bundle_root}"
job_file=${ROBOTARM_JOB_FILE:-${bundle_root}/job.env}
runtime_dir=${ROBOTARM_RUNTIME_DIR:-${bundle_root}/.runtime}
train_pid_file=${runtime_dir}/train.pid
train_log=${runtime_dir}/train.log
monitor_pid_file=${runtime_dir}/monitor.pid
monitor_log=${runtime_dir}/monitor.log
mkdir -p "${runtime_dir}"

usage() {
  cat <<'EOF'
Usage: bash scripts/cloud_vla/service.sh COMMAND

Training commands:
  start          Start a new formal run in the background.
  resume         Resume from RESUME_CONFIG in the background; STEPS is the total target.
  status         Show the training process status.
  logs           Follow the training log (Ctrl-C only exits tail).
  stop           Send SIGINT and wait up to 30 seconds; preserves completed checkpoints.
  stop-force     Send SIGTERM if graceful stop did not work.

Monitor commands:
  monitor-start  Start the training web monitor in the background.
  monitor-status Show monitor status and URL.
  monitor-logs   Follow monitor logs.
  monitor-stop   Stop the web monitor.
EOF
}

read_pid() {
  local file=$1
  [[ -f ${file} ]] && tr -d '[:space:]' < "${file}"
}

alive() {
  local pid=${1:-}
  [[ ${pid} =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start_background() {
  local pid_file=$1 log_file=$2 label=$3
  shift 3
  local old_pid
  old_pid=$(read_pid "${pid_file}" || true)
  if alive "${old_pid}"; then
    echo "${label} is already running (PID ${old_pid})." >&2
    exit 1
  fi
  rm -f -- "${pid_file}"
  nohup setsid "$@" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  sleep 1
  if ! alive "${pid}"; then
    echo "${label} exited during startup. Last log lines:" >&2
    tail -n 60 "${log_file}" >&2 || true
    rm -f -- "${pid_file}"
    exit 1
  fi
  echo "${label} started: PID ${pid}"
  echo "Log: ${log_file}"
}

stop_group() {
  local pid_file=$1 label=$2 signal=${3:-INT}
  local pid
  pid=$(read_pid "${pid_file}" || true)
  if ! alive "${pid}"; then
    echo "${label} is not running."
    rm -f -- "${pid_file}"
    return 0
  fi
  kill -s "${signal}" -- "-${pid}" 2>/dev/null || kill -s "${signal}" "${pid}"
  for _ in $(seq 1 30); do
    if ! alive "${pid}"; then
      rm -f -- "${pid_file}"
      echo "${label} stopped."
      return 0
    fi
    sleep 1
  done
  echo "${label} still has PID ${pid}; inspect logs before using stop-force." >&2
  return 1
}

show_status() {
  local pid_file=$1 label=$2
  local pid
  pid=$(read_pid "${pid_file}" || true)
  if alive "${pid}"; then
    echo "${label}: running (PID ${pid})"
  else
    echo "${label}: stopped"
  fi
}

load_job() {
  if [[ ! -f ${job_file} ]]; then
    echo "Missing ${job_file}" >&2
    exit 1
  fi
  set -a
  source "${job_file}"
  set +a
}

command=${1:-}
case "${command}" in
  start)
    start_background "${train_pid_file}" "${train_log}" "training" \
      "${bundle_root}/scripts/cloud_vla/run_train.sh" --execute
    ;;
  resume)
    load_job
    if [[ -z ${RESUME_CONFIG:-} || ! -f ${RESUME_CONFIG} ]]; then
      echo "Set RESUME_CONFIG in job.env to an existing pretrained_model/train_config.json." >&2
      exit 2
    fi
    source "${bundle_root}/scripts/cloud_vla/cloud_env.sh"
    start_background "${train_pid_file}" "${train_log}" "training resume" \
      lerobot-train --config_path="${RESUME_CONFIG}" --resume=true --steps="${STEPS}"
    ;;
  status)
    show_status "${train_pid_file}" "training"
    ;;
  logs)
    touch "${train_log}"
    tail -n 100 -f "${train_log}"
    ;;
  stop)
    stop_group "${train_pid_file}" "training" INT
    ;;
  stop-force)
    stop_group "${train_pid_file}" "training" TERM
    ;;
  monitor-start)
    load_job
    source "${bundle_root}/scripts/cloud_vla/cloud_env.sh"
    monitor_port=${MONITOR_PORT:-8420}
    start_background "${monitor_pid_file}" "${monitor_log}" "monitor" \
      python "${bundle_root}/scripts/train_monitor.py" \
      --run-dir "${OUTPUT_DIR}" --port "${monitor_port}" --interval 2
    echo "URL: http://127.0.0.1:${monitor_port}"
    ;;
  monitor-status)
    load_job
    show_status "${monitor_pid_file}" "monitor"
    echo "URL: http://127.0.0.1:${MONITOR_PORT:-8420}"
    ;;
  monitor-logs)
    touch "${monitor_log}"
    tail -n 100 -f "${monitor_log}"
    ;;
  monitor-stop)
    stop_group "${monitor_pid_file}" "monitor" INT
    ;;
  *)
    usage
    [[ -n ${command} ]] && exit 2 || exit 0
    ;;
esac
