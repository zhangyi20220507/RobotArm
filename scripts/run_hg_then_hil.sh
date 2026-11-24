#!/usr/bin/env bash
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${workspace}"
exec python -m robotarm.modeling.hil_serl.cli pipeline "$@"
