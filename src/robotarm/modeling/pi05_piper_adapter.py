"""Offline Pi0.5 camera and Piper action adapter. Never sends CAN frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping

import numpy as np

from ..core.robot.safety import safe_joint_bounds


CAMERA_KEYS = {
    "lerobot": {
        "global": "observation.images.base_0_rgb",
        "left_wrist": "observation.images.left_wrist_0_rgb",
        "right_wrist": "observation.images.right_wrist_0_rgb",
    },
    "robodojo": {
        "global": "observation.images.cam_high",
        "left_wrist": "observation.images.cam_left_wrist",
        "right_wrist": "observation.images.cam_right_wrist",
    },
}


@dataclass(frozen=True)
class ActionContract:
    arms: tuple[str, ...]
    joint_mode: str  # absolute or delta_from_observation
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.arms not in (("left",), ("right",), ("left", "right")):
            raise ValueError("arms must be left, right, or left+right in order")
        if self.joint_mode not in ("absolute", "delta_from_observation"):
            raise ValueError("unsupported joint action mode")


@dataclass(frozen=True)
class SafetyLimits:
    max_camera_age_s: float = 0.10
    max_feedback_age_s: float = 0.20
    max_joint_step_rad: float = math.radians(0.25)
    max_joint_lead_rad: float = math.radians(0.5)
    max_gripper_step_m: float = 0.0005
    max_gripper_lead_m: float = 0.001
    gripper_max_m: float = 0.1


@dataclass(frozen=True)
class ArmFeedback:
    state: np.ndarray  # J1-J6 radians; gripper width metres
    captured_at: float  # monotonic clock


@dataclass(frozen=True)
class PiperTarget:
    joints_rad: tuple[float, ...]
    gripper_width_m: float

    def can_calls(self) -> tuple[tuple[str, object], ...]:
        """CAN SDK method names and arguments; this function performs no I/O."""
        return (("move_js", list(self.joints_rad)), ("move_gripper_m", self.gripper_width_m))


def map_camera_frames(
    frames: Mapping[str, tuple[np.ndarray, float]],
    *,
    profile: str = "lerobot",
    require_wrists: bool = False,
    max_age_s: float = 0.10,
    now: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert BGR HWC uint8 frames to RGB CHW float32 and map role names."""
    if profile not in CAMERA_KEYS or not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("invalid camera profile or age limit")
    now = time.monotonic() if now is None else now
    required = ("global", "left_wrist", "right_wrist") if require_wrists else ("global",)
    missing = set(required) - frames.keys()
    if missing:
        raise RuntimeError(f"missing required camera roles: {sorted(missing)}")
    result = {}
    for role in CAMERA_KEYS[profile]:
        if role not in frames:
            continue
        bgr, captured_at = frames[role]
        age = now - captured_at
        if not math.isfinite(age) or not 0 <= age <= max_age_s:
            raise RuntimeError(f"{role} camera frame stale or future-dated: {age:.3f}s")
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RuntimeError(f"{role} camera must be HWC uint8 BGR")
        result[CAMERA_KEYS[profile][role]] = (
            bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        )
    return result


def _state(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise RuntimeError(f"{name} must be finite 7D Piper state")
    return array


def decode_and_validate(
    action: np.ndarray,
    contract: ActionContract,
    feedback: Mapping[str, ArmFeedback],
    reference: Mapping[str, np.ndarray],
    previous: Mapping[str, np.ndarray],
    configs: Mapping[str, dict],
    *,
    limits: SafetyLimits = SafetyLimits(),
    now: float | None = None,
) -> dict[str, PiperTarget]:
    """Return physical CAN targets or reject the whole step, including both arms."""
    if not contract.calibrated:
        raise RuntimeError("uncalibrated/base Pi0.5 actions cannot become CAN targets")
    raw = np.asarray(action, dtype=np.float64)
    if raw.shape != (7 * len(contract.arms),) or not np.isfinite(raw).all():
        raise RuntimeError(f"expected finite {7 * len(contract.arms)}D calibrated action")
    if any(not math.isfinite(v) or v <= 0 for v in vars(limits).values()):
        raise ValueError("all safety limits must be positive and finite")
    now = time.monotonic() if now is None else now
    targets: dict[str, PiperTarget] = {}
    for index, arm in enumerate(contract.arms):
        if arm not in feedback or arm not in reference or arm not in previous or arm not in configs:
            raise RuntimeError(f"missing {arm} feedback, reference, previous target, or limits")
        measured = _state(feedback[arm].state, f"{arm} feedback")
        observed = _state(reference[arm], f"{arm} reference")
        prior = _state(previous[arm], f"{arm} previous target")
        age = now - feedback[arm].captured_at
        if not math.isfinite(age) or not 0 <= age <= limits.max_feedback_age_s:
            raise RuntimeError(f"{arm} CAN feedback stale or future-dated: {age:.3f}s")
        segment = raw[index * 7 : (index + 1) * 7]
        joints = segment[:6] + observed[:6] if contract.joint_mode == "delta_from_observation" else segment[:6]
        width = float(segment[6])  # gripper is absolute in both contracts
        bounds = safe_joint_bounds(configs[arm], margin_degrees=1.0)
        for j, (value, before, old, (low, high)) in enumerate(
            zip(joints, measured[:6], prior[:6], bounds), start=1
        ):
            if not low <= value <= high:
                raise RuntimeError(f"{arm} J{j} target outside software joint limits")
            if abs(value - old) > limits.max_joint_step_rad:
                raise RuntimeError(f"{arm} J{j} target step exceeds limit")
            if abs(value - before) > limits.max_joint_lead_rad:
                raise RuntimeError(f"{arm} J{j} following error exceeds limit")
        if not 0 <= width <= limits.gripper_max_m:
            raise RuntimeError(f"{arm} gripper width outside limits")
        if abs(width - prior[6]) > limits.max_gripper_step_m:
            raise RuntimeError(f"{arm} gripper target step exceeds limit")
        if abs(width - measured[6]) > limits.max_gripper_lead_m:
            raise RuntimeError(f"{arm} gripper following error exceeds limit")
        targets[arm] = PiperTarget(tuple(float(x) for x in joints), width)
    return targets
