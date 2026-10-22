#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
conda_exe=${CONDA_EXE:-/home/dm/miniforge3/bin/conda}
env_name=${ROBOTARM_ENV_NAME:-robotarm}
env_prefix=$($conda_exe info --base)/envs/${env_name}

if [[ -e ${env_prefix} ]]; then
  echo "Environment already exists: ${env_prefix}" >&2
  echo "Remove it explicitly before recreating it." >&2
  exit 1
fi

"${conda_exe}" env create -n "${env_name}" -f "${repo_root}/environment.yml"
python_bin="${env_prefix}/bin/python"

"${python_bin}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
# Install TorchAO from the same CUDA wheel channel as PyTorch.  The PyPI
# torchao wheel currently links against CUDA 13 and emits loader warnings in
# this CUDA 12.8 environment even when the policy does not use quantization.
"${python_bin}" -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torchao==0.17.0+cu128
"${python_bin}" -m pip install \
  -c "${repo_root}/requirements/unified-constraints.txt" \
  -e "${repo_root}[hardware,modeling,simulation,xr]"

# Placo 0.9.23's current cmeel-boost metadata selects NumPy >=2.3 on Python
# 3.12, while LeRobot 0.6.1 requires NumPy <2.3. The Placo binaries used here
# have been runtime-tested with 2.2.6, so install the native stack first and
# restore the LeRobot-compatible NumPy afterwards.
"${python_bin}" -m pip install placo==0.9.23
"${python_bin}" -m pip install --no-deps numpy==2.2.6
"${python_bin}" -m pip install --no-build-isolation "${repo_root}/vendor/xrobotoolkit_sdk"

mkdir -p "${env_prefix}/etc/conda/activate.d" "${env_prefix}/etc/conda/deactivate.d"
install -m 0644 "${repo_root}/config/conda/activate.d/robotarm.sh" \
  "${env_prefix}/etc/conda/activate.d/robotarm.sh"
install -m 0644 "${repo_root}/config/conda/deactivate.d/robotarm.sh" \
  "${env_prefix}/etc/conda/deactivate.d/robotarm.sh"

echo "Created ${env_name}. Activate it with: conda activate ${env_name}"

