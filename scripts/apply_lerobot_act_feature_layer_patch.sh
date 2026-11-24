#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
mode=apply

usage() {
  echo "Usage: $0 [--apply|--check] [--python /path/to/python]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      mode=apply
      shift
      ;;
    --check)
      mode=check
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      python_bin=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

readarray -t install_info < <("${python_bin}" - <<'PY'
import importlib.metadata
from pathlib import Path

import lerobot

print(importlib.metadata.version("lerobot"))
print(Path(lerobot.__file__).resolve().parent.parent)
PY
)

lerobot_version=${install_info[0]}
site_packages=${install_info[1]}
config_file=${site_packages}/lerobot/policies/act/configuration_act.py
model_file=${site_packages}/lerobot/policies/act/modeling_act.py
patch_file=${repo_root}/patches/lerobot-0.6.1-act-vision-feature-layer.patch

if [[ ${lerobot_version} != 0.6.1 ]]; then
  echo "Refusing to patch lerobot==${lerobot_version}; this patch is pinned to lerobot==0.6.1." >&2
  exit 1
fi

is_patched=false
if grep -q '^    vision_feature_layer: str = "layer4"$' "${config_file}" && \
   grep -q 'feature_layer = config\.vision_feature_layer' "${model_file}"; then
  is_patched=true
fi

if [[ ${mode} == check ]]; then
  if [[ ${is_patched} != true ]]; then
    echo "LeRobot ACT feature-layer patch is not applied to ${site_packages}." >&2
    exit 1
  fi
  exec "${python_bin}" "${repo_root}/scripts/verify_lerobot_act_feature_layer.py"
fi

if [[ ${is_patched} == true ]]; then
  echo "LeRobot ACT feature-layer patch is already applied: ${site_packages}"
  exec "${python_bin}" "${repo_root}/scripts/verify_lerobot_act_feature_layer.py"
fi

config_sha=$(sha256sum "${config_file}" | awk '{print $1}')
model_sha=$(sha256sum "${model_file}" | awk '{print $1}')
expected_config_sha=35a60c0dd3c811b137b5eb035bdbd40d40e2c24ba0cda61756052e6e1c829282
expected_model_sha=a6c1a49c5daafa513b9abee9b986205d327e25c70ed13623ab2e6c0ab6b1f577

if [[ ${config_sha} != "${expected_config_sha}" || ${model_sha} != "${expected_model_sha}" ]]; then
  echo "Refusing to patch: LeRobot ACT source does not match the tested 0.6.1 files." >&2
  echo "configuration_act.py sha256=${config_sha}" >&2
  echo "modeling_act.py sha256=${model_sha}" >&2
  echo "Expected ${expected_config_sha} and ${expected_model_sha}." >&2
  exit 1
fi

if command -v patch >/dev/null 2>&1; then
  patch --batch --forward -p1 -d "${site_packages}" < "${patch_file}"
elif command -v git >/dev/null 2>&1; then
  git apply --unsafe-paths --directory="${site_packages}" "${patch_file}"
else
  echo "Neither 'patch' nor 'git' is installed; install one and rerun this script." >&2
  exit 1
fi

echo "Applied LeRobot ACT feature-layer patch to ${site_packages}"
exec "${python_bin}" "${repo_root}/scripts/verify_lerobot_act_feature_layer.py"
