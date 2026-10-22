"""基于 Placo 的 Piper 本地逆解与关节限位。

迁移自 teleop_piper_xr.py 的 IK 部分；依赖 placo/numpy/scipy，
仅由 XR 运行时在执行期导入。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import placo
from scipy.spatial.transform import Rotation

from ...paths import PIPER_URDF

PIPER_NOMINAL_JOINT_MIN = np.array(
    [-2.6179938, 0.0, -2.9670597, -1.7453292, -1.2217304, -2.0943951]
)
PIPER_NOMINAL_JOINT_MAX = np.array(
    [2.6179938, 3.1415926, 0.0, 1.7453292, 1.2217304, 2.0943951]
)
# Match the proven policy in the gamepad runtime: measured J2/J3 may have up
# to 3 degrees of zero-offset, while commands keep a 1-degree margin from the
# resulting bounds.  This admits about +/-2 degrees around the nominal J2/J3
# zero instead of forcing a measured joint abruptly to zero.
_COMMAND_MARGIN = math.radians(1.0)
_J23_ZERO_OFFSET = math.radians(3.0)
PIPER_JOINT_MIN = PIPER_NOMINAL_JOINT_MIN + _COMMAND_MARGIN
PIPER_JOINT_MAX = PIPER_NOMINAL_JOINT_MAX - _COMMAND_MARGIN
PIPER_JOINT_MIN[1:3] -= _J23_ZERO_OFFSET
PIPER_JOINT_MAX[1:3] += _J23_ZERO_OFFSET


@dataclass(frozen=True)
class IKDiagnostics:
    solve_ms: float
    position_error_m: float
    orientation_error_rad: float
    original_position_error_m: float
    target_scale: float
    attempts: int
    position_condition: float
    pose_condition: float


@dataclass(frozen=True)
class IKSolveResult:
    joints: np.ndarray
    accepted_target: np.ndarray
    diagnostics: IKDiagnostics


class PiperIK:
    """Local Placo IK for the standard six-axis Piper model."""

    def __init__(self, dt: float, orientation_enabled: bool, orientation_weight: float = 0.00001):
        self.robot = placo.RobotWrapper(str(PIPER_URDF))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.dt = dt
        self.orientation_enabled = orientation_enabled

        self.robot.state.q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.robot.update_kinematics()

        self.ee_task = self.solver.add_position_task("link6", np.zeros(3))
        self.ee_task.configure("piper_position", "soft", 1.0)
        self.orientation_task = self.solver.add_orientation_task("link6", np.eye(3))
        # Position takes priority near wrist singularities. Translation mode
        # prefers the reference attitude but may relax it to allow translation.
        self.orientation_task.configure(
            "piper_orientation", "soft", 1.0 if orientation_enabled else orientation_weight
        )

        self.joints_task = self.solver.add_joints_task()
        self.joints_task.configure("joint_regularization", "soft", 1.0e-4)
        # Null-space posture target.  Following the IK seed only resists change
        # and lets repeated motions ratchet the arm into unrelated postures;
        # a fixed anchor (captured at engagement) restores the starting posture
        # instead.  None keeps the legacy seed-following behaviour.
        self.posture_anchor: np.ndarray | None = None

    def set_posture_anchor(self, joints: np.ndarray | None) -> None:
        """Fix (or clear) the posture the null-space regularizer pulls toward."""
        self.posture_anchor = (
            None if joints is None else np.asarray(joints, dtype=float).copy()
        )

    def set_state(self, joints: np.ndarray) -> None:
        self.robot.state.q[7:13] = joints
        self.robot.update_kinematics()

    def flange_pose(self, joints: np.ndarray) -> np.ndarray:
        self.set_state(joints)
        transform = self.robot.get_T_world_frame("link6")
        return np.concatenate(
            [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")]
        )

    def solve(self, joints: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        """Solve one target without residual-based backtracking (compatibility API)."""
        return self._solve_once(joints, target_pose)

    def _solve_once(self, joints: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        self.set_state(joints)
        posture = joints if self.posture_anchor is None else self.posture_anchor
        self.joints_task.set_joints({f"joint{i + 1}": float(posture[i]) for i in range(6)})
        self.ee_task.target_world = target_pose[:3]
        self.orientation_task.R_world_frame = Rotation.from_euler("xyz", target_pose[3:]).as_matrix()

        # Converge the nonlinear pose target instead of accepting one Jacobian step.
        # The physical command still passes the Cartesian/joint speed limits.
        for _ in range(4):
            self.solver.solve(True)
            self.robot.update_kinematics()
            transform = self.robot.get_T_world_frame("link6")
            position_error = np.linalg.norm(transform[:3, 3] - target_pose[:3])
            rotation_error = (Rotation.from_matrix(transform[:3, :3]) *
                              Rotation.from_euler("xyz", target_pose[3:]).inv()).magnitude()
            if position_error <= 0.0001 and (not self.orientation_enabled or rotation_error <= math.radians(0.1)):
                break
        solution = self.robot.state.q[7:13].copy()
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Placo IK returned non-finite joint values")
        if np.any(solution < PIPER_JOINT_MIN - 1.0e-6) or np.any(
            solution > PIPER_JOINT_MAX + 1.0e-6
        ):
            raise RuntimeError("Placo IK solution exceeds Piper joint limits")
        return solution

    @staticmethod
    def _orientation_error(left: np.ndarray, right: np.ndarray) -> float:
        left_rotation = Rotation.from_euler("xyz", left)
        right_rotation = Rotation.from_euler("xyz", right)
        return float((left_rotation * right_rotation.inv()).magnitude())

    @staticmethod
    def _interpolate_pose(start: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
        result = np.asarray(start, dtype=float).copy()
        result[:3] += (np.asarray(target[:3]) - result[:3]) * scale
        start_rotation = Rotation.from_euler("xyz", start[3:])
        target_rotation = Rotation.from_euler("xyz", target[3:])
        rotation_delta = (target_rotation * start_rotation.inv()).as_rotvec()
        result[3:] = (
            Rotation.from_rotvec(rotation_delta * scale) * start_rotation
        ).as_euler("xyz")
        return result

    def condition_numbers(self, joints: np.ndarray) -> tuple[float, float]:
        """Return translational and 0.1 m-scaled full-pose Jacobian conditions."""
        self.set_state(joints)
        jacobian = np.asarray(
            self.robot.frame_jacobian("link6", "world"), dtype=float
        )[:, -6:]
        position_condition = float(np.linalg.cond(jacobian[:3]))
        # Scaling angular rows by a characteristic 0.1 m makes the mixed-unit
        # condition useful for trends without pretending radians equal metres.
        pose_jacobian = np.vstack((jacobian[:3], jacobian[3:] * 0.1))
        pose_condition = float(np.linalg.cond(pose_jacobian))
        return position_condition, pose_condition

    def solve_adaptive(
        self,
        seed_joints: np.ndarray,
        target_pose: np.ndarray,
        *,
        position_tolerance: float = 0.0008,
        orientation_tolerance: float = math.radians(1.0),
        scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
    ) -> IKSolveResult:
        """Solve with residual validation and progressively smaller target steps."""
        seed = np.asarray(seed_joints, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        start_pose = self.flange_pose(seed)
        started = time.perf_counter()
        failures: list[str] = []

        for attempt, scale in enumerate(scales, start=1):
            candidate = self._interpolate_pose(start_pose, target, scale)
            try:
                solution = self._solve_once(seed, candidate)
                achieved = self.flange_pose(solution)
            except RuntimeError as exc:
                failures.append(f"{scale:.3f}x: {exc}")
                continue
            position_error = float(np.linalg.norm(achieved[:3] - candidate[:3]))
            orientation_error = self._orientation_error(achieved[3:], candidate[3:])
            orientation_ok = (
                not self.orientation_enabled
                or orientation_error <= orientation_tolerance
            )
            if position_error <= position_tolerance and orientation_ok:
                position_condition, pose_condition = self.condition_numbers(solution)
                diagnostics = IKDiagnostics(
                    solve_ms=(time.perf_counter() - started) * 1000.0,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    original_position_error_m=float(
                        np.linalg.norm(achieved[:3] - target[:3])
                    ),
                    target_scale=scale,
                    attempts=attempt,
                    position_condition=position_condition,
                    pose_condition=pose_condition,
                )
                return IKSolveResult(solution, candidate, diagnostics)
            failures.append(
                f"{scale:.3f}x: residual={position_error * 1000.0:.2f}mm/"
                f"{math.degrees(orientation_error):.2f}deg"
            )

        raise RuntimeError(
            "Placo IK residual remained above tolerance after adaptive shrinking: "
            + "; ".join(failures)
        )


def joint_limit_violations(joints: np.ndarray, tolerance: float) -> list[str]:
    violations = []
    for index, value in enumerate(joints):
        if value < PIPER_NOMINAL_JOINT_MIN[index] - tolerance:
            violations.append(
                f"J{index + 1}={value:.4f} < nominal min "
                f"{PIPER_NOMINAL_JOINT_MIN[index]:.4f} rad"
            )
        elif value > PIPER_NOMINAL_JOINT_MAX[index] + tolerance:
            violations.append(
                f"J{index + 1}={value:.4f} > nominal max "
                f"{PIPER_NOMINAL_JOINT_MAX[index]:.4f} rad"
            )
    return violations
