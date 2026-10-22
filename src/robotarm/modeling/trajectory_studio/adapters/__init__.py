from .act import LeRobotPolicyAdapter
from .base import ModelAdapter

ADAPTERS: dict[str, ModelAdapter] = {
    "act": LeRobotPolicyAdapter("act", "LeRobot ACT", "act"),
    "diffusion": LeRobotPolicyAdapter(
        "diffusion", "LeRobot Diffusion Policy", "diffusion"
    ),
    "smolvla": LeRobotPolicyAdapter("smolvla", "LeRobot SmolVLA", "smolvla"),
}

__all__ = ["ADAPTERS", "ModelAdapter"]
