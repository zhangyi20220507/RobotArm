#!/usr/bin/env bash
set -euo pipefail

# Reproducible training-only environment for a local workstation or cloud GPU host.
# It intentionally omits camera/CAN/XR/simulation dependencies used by the full environment.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
conda_exe=${CONDA_EXE:-}
env_name=${ROBOTARM_ACT_ENV_NAME:-robotarm-act}
pytorch_index_url=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
torchao_version=${TORCHAO_VERSION:-0.17.0+cu128}

if [[ -z ${conda_exe} ]]; then
  conda_exe=$(command -v conda || true)
fi
if [[ -z ${conda_exe} ]]; then
  echo "Conda was not found. Install Miniforge/Miniconda or set CONDA_EXE explicitly." >&2
  exit 1
fi
if "${conda_exe}" run -n "${env_name}" true >/dev/null 2>&1; then
  echo "Environment already exists: ${env_name}" >&2
  echo "Use a new ROBOTARM_ACT_ENV_NAME or remove the old environment explicitly." >&2
  exit 1
fi

"${conda_exe}" env create -n "${env_name}" -f "${repo_root}/environment.yml"
python_bin=$("${conda_exe}" run -n "${env_name}" python -c 'import sys; print(sys.executable)')

"${python_bin}" -m pip install \
  --index-url "${pytorch_index_url}" \
  torch==2.11.0 torchvision==0.26.0
"${python_bin}" -m pip install --no-deps \
  --index-url "${pytorch_index_url}" \
  "torchao==${torchao_version}"
"${python_bin}" -m pip install \
  -c "${repo_root}/requirements/unified-constraints.txt" \
  -e "${repo_root}[modeling]"

"${repo_root}/scripts/apply_lerobot_act_feature_layer_patch.sh" \
  --python "${python_bin}" --apply

"${python_bin}" -m pip check
echo "Created ${env_name} with the ACT layer3/layer4 extension."
echo "Activate it with: conda activate ${env_name}"
