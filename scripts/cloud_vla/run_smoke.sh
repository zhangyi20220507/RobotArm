#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
job_file=${ROBOTARM_JOB_FILE:-${bundle_root}/job.env}
if [[ ! -f ${job_file} ]]; then
  echo "Missing ${job_file}" >&2
  exit 1
fi
set -a
source "${job_file}"
set +a
export OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${OUTPUT_DIR}_smoke}"
temp_job=$(mktemp "${TMPDIR:-/tmp}/robotarm-vla-smoke.XXXXXX")
trap 'rm -f -- "${temp_job}"' EXIT
cp -- "${job_file}" "${temp_job}"
printf '\nOUTPUT_DIR=%q\n' "${OUTPUT_DIR}" >> "${temp_job}"
ROBOTARM_JOB_FILE="${temp_job}" exec "${bundle_root}/scripts/cloud_vla/run_train.sh" --smoke "$@"
