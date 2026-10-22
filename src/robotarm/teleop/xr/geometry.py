"""PICO 跟踪坐标系 → Piper 基坐标系的变换、工作空间限位与目标构建。

迁移自 teleop_piper_xr.py 的几何部分；依赖 numpy/scipy，
仅由 XR 运行时在执行期导入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# PICO tracking coordinates -> Piper base coordinates. This is the same basis
# conversion used by the XRoboToolkit sample controllers.
R_XR_TO_PIPER = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

# A single horizontal axis cannot be flipped by --xr-yaw-deg: every yaw keeps
# the mapping a proper rotation.  An operator facing the robot can therefore
# see left-right (or forward-back) reversed while the other axes are correct,
# which needs a reflection instead.
MIRROR_FLIPS = {
    "none": np.eye(3),
    "lr": np.diag([1.0, -1.0, 1.0]),
    "fb": np.diag([-1.0, 1.0, 1.0]),
}


def mapping_matrix(yaw_degrees: float = 0.0, mirror: str = "none") -> np.ndarray:
    """XR tracking axes -> Piper base axes, with the yaw calibration and mirror."""
    try:
        flip = MIRROR_FLIPS[mirror]
    except KeyError as exc:
        raise ValueError(f"unknown xr mirror mode: {mirror}") from exc
    return flip @ Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix() @ R_XR_TO_PIPER


@dataclass
class Limits:
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    max_relative_translation: float
    max_step_translation: float
    max_step_rotation: float


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def clamp_vector_step(current: np.ndarray, requested: np.ndarray, max_norm: float) -> np.ndarray:
    delta = requested - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm or norm == 0.0:
        return requested
    return current + delta * (max_norm / norm)


def valid_xr_pose(pose: np.ndarray) -> bool:
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return False
    quat_norm = float(np.linalg.norm(pose[3:]))
    return quat_norm > 0.5


def xr_rotation(pose: np.ndarray) -> Rotation:
    # XRoboToolkit uses [x, y, z, qx, qy, qz, qw].
    return Rotation.from_quat(pose[3:7])


def build_target(
    xr_reference: np.ndarray,
    xr_current: np.ndarray,
    flange_reference: np.ndarray,
    scale: float,
    orientation_enabled: bool,
    limits: Limits,
    yaw_degrees: float = 0.0,
    mirror: str = "none",
) -> np.ndarray:
    delta_xr = xr_current[:3] - xr_reference[:3]
    mapping = mapping_matrix(yaw_degrees, mirror)
    delta_base = mapping @ delta_xr * scale
    delta_norm = float(np.linalg.norm(delta_base))
    if delta_norm > limits.max_relative_translation:
        delta_base *= limits.max_relative_translation / delta_norm

    target = flange_reference.copy()
    # If feedback starts slightly outside the configured workspace, do not
    # jump immediately to the boundary or refuse the clutch.  Expand only the
    # boundary needed to include the clutch reference: motion may recover
    # inward, but cannot move farther out than the measured starting pose.
    recovery_min = np.minimum(limits.xyz_min, flange_reference[:3])
    recovery_max = np.maximum(limits.xyz_max, flange_reference[:3])
    target[:3] = np.clip(
        flange_reference[:3] + delta_base,
        recovery_min,
        recovery_max,
    )

    if orientation_enabled:
        ref_matrix = xr_rotation(xr_reference).as_matrix()
        current_matrix = xr_rotation(xr_current).as_matrix()
        delta_xr_matrix = current_matrix @ ref_matrix.T
        delta_base_matrix = mapping @ delta_xr_matrix @ mapping.T
        flange_matrix = Rotation.from_euler("xyz", flange_reference[3:]).as_matrix()
        target[3:] = Rotation.from_matrix(delta_base_matrix @ flange_matrix).as_euler("xyz")
        target[3:] = wrap_angles(target[3:])
        target[4] = np.clip(target[4], -np.pi / 2.0, np.pi / 2.0)

    return target


def clamp_rotation_step(current: np.ndarray, requested: np.ndarray, max_angle: float) -> np.ndarray:
    """Limit the shortest SO(3) rotation, including near Euler singularities."""
    start = Rotation.from_euler("xyz", current)
    delta = (Rotation.from_euler("xyz", requested) * start.inv()).as_rotvec()
    angle = float(np.linalg.norm(delta))
    if angle > max_angle:
        delta *= max_angle / angle
    return (Rotation.from_rotvec(delta) * start).as_euler("xyz")


class TranslationTracker:
    """Incremental hand target with anti-windup and fixed clutch boundaries.

    Discard motion beyond a boundary/lead cap instead of queuing it for later.
    The original clutch position continues to define the relative travel limit.
    """

    def __init__(self, hand, flange, scale, limits, max_lead, yaw_degrees=0.0, mirror="none"):
        self.previous_hand = np.asarray(hand, dtype=float).copy()
        self.reference = np.asarray(flange, dtype=float).copy()
        self.target = self.reference.copy()
        self.mapping = mapping_matrix(yaw_degrees, mirror)
        self.scale = scale
        self.limits = limits
        self.max_lead = max_lead
        self.minimum = np.minimum(limits.xyz_min, self.reference)
        self.maximum = np.maximum(limits.xyz_max, self.reference)

    def update(self, hand, feedback):
        hand = np.asarray(hand, dtype=float)
        delta = self.mapping @ (hand - self.previous_hand) * self.scale
        self.previous_hand = hand.copy()
        requested = self.target + delta
        reasons = []
        bounded = clamp_vector_step(self.reference, requested, self.limits.max_relative_translation)
        if np.linalg.norm(bounded - requested) > 1e-8:
            reasons.append("RELATIVE_LIMIT: 松开握持键，收回手后重新握住")
        workspace = np.clip(bounded, self.minimum, self.maximum)
        if np.linalg.norm(workspace - bounded) > 1e-8:
            reasons.append("WORKSPACE_LIMIT: 已到工作空间边界，请向内移动")
        limited = clamp_vector_step(np.asarray(feedback), workspace, self.max_lead)
        if np.linalg.norm(limited - workspace) > 1e-8:
            reasons.append("FOLLOW_LIMIT: 机械臂跟随受限，已丢弃超前位移，请放慢手速")
        # Joint feedback can lie outside the configured box; never expand the
        # fixed clutch bounds just because feedback drifts during engagement.
        limited = np.clip(limited, self.minimum, self.maximum)
        self.target = clamp_vector_step(self.reference, limited, self.limits.max_relative_translation)
        return self.target.copy(), reasons
