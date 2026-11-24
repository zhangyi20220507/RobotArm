#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${ROBOTARM_PYTHON:-python3.12}
venv_dir=${ROBOTARM_VENV_DIR:-${bundle_root}/.venv-cloud}
pytorch_index_url=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; install a compatible NVIDIA driver before creating the environment." >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "${python_bin} not found; install Python 3.12 and python3.12-venv." >&2
  exit 1
fi
if [[ -e ${venv_dir} ]]; then
  echo "Environment already exists: ${venv_dir}" >&2
  echo "Reuse it, or remove that exact directory explicitly before rebuilding." >&2
  exit 1
fi

"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install \
  --index-url "${pytorch_index_url}" \
  torch==2.11.0 torchvision==0.26.0
"${venv_dir}/bin/python" -m pip install --no-deps \
  --index-url "${pytorch_index_url}" \
  torchao==0.17.0+cu128
"${venv_dir}/bin/python" -m pip install \
  -c "${bundle_root}/requirements/unified-constraints.txt" \
  -e "${bundle_root}[cloud-training]"
"${venv_dir}/bin/python" -m pip check

source "${venv_dir}/bin/activate"
python "${bundle_root}/scripts/cloud_vla/preflight.py" --environment-only
echo "Cloud training environment ready: ${venv_dir}"
echo "For each new shell: source scripts/cloud_vla/cloud_env.sh"
