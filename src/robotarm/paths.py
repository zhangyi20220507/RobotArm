from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 数据与训练工件在代码树之外，可通过环境变量重定向。
DATA_ROOT = Path(os.environ.get("ROBOTARM_DATA_ROOT", WORKSPACE_ROOT / "data")).expanduser()
ARTIFACTS_ROOT = Path(
    os.environ.get("ROBOTARM_ARTIFACTS_ROOT", WORKSPACE_ROOT / "artifacts")
).expanduser()

RAW_DATA_ROOT = DATA_ROOT / "raw"
PIPER_RAW_ROOT = RAW_DATA_ROOT / "piper_raw"
LEROBOT_DATA_ROOT = DATA_ROOT / "lerobot"

CHECKPOINT_ROOT = ARTIFACTS_ROOT / "checkpoints"
TRAINING_LOG_ROOT = ARTIFACTS_ROOT / "logs"
PRETRAINED_ROOT = ARTIFACTS_ROOT / "pretrained"
SIMULATION_DATA_ROOT = DATA_ROOT / "simulation"
SIMULATION_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "simulation"

# Piper 本地 SDK（pyAgxArm）位置：vendored 第三方源码。
VENDOR_ROOT = WORKSPACE_ROOT / "vendor"
PYAGXARM_ROOT = VENDOR_ROOT / "pyAgxArm"

# XR 遥操作使用的 URDF 运动学模型（随包分发）。
PIPER_URDF = Path(__file__).resolve().parent / "assets" / "piper_kinematics.urdf"
PIPER_MANISKILL_URDF = (
    Path(__file__).resolve().parent / "assets" / "piper" / "maniskill" / "piper_description.urdf"
)

# 模型审批配置目录（随包分发）。
MODELING_CONFIGS_ROOT = Path(__file__).resolve().parent / "modeling" / "configs"

def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
