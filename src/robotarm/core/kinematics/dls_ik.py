"""本地逆解模块：提供纯 Python 的阻尼最小二乘（DLS）逆运动学求解器。

调用方需提供正运动学函数、目标位姿、当前关节角和关节边界。

迁移自 piper_ctrl/control_scripts/src/dls_ik.py。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence


class IKError(RuntimeError):
    """Raised when the local numerical IK solver cannot find a safe solution."""


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sy],
        [-sp, cp * sr, cp * cr],
    ]


def _transpose3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matmul3(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _rotation_vector(matrix: Sequence[Sequence[float]]) -> list[float]:
    cosine = max(-1.0, min(1.0, (sum(matrix[i][i] for i in range(3)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    skew = [
        matrix[2][1] - matrix[1][2],
        matrix[0][2] - matrix[2][0],
        matrix[1][0] - matrix[0][1],
    ]
    if angle < 1e-7:
        return [value / 2.0 for value in skew]
    sine = math.sin(angle)
    if abs(sine) < 1e-7:
        raise IKError("orientation difference is too close to 180 degrees")
    scale = angle / (2.0 * sine)
    return [value * scale for value in skew]


def pose_error(target: Sequence[float], current: Sequence[float]) -> list[float]:
    """Return world-frame position and rotation-vector error."""
    target_rotation = _rpy_matrix(*target[3:6])
    current_rotation = _rpy_matrix(*current[3:6])
    rotation_error = _matmul3(target_rotation, _transpose3(current_rotation))
    return [
        target[0] - current[0],
        target[1] - current[1],
        target[2] - current[2],
        *_rotation_vector(rotation_error),
    ]


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with pivoted Gaussian elimination."""
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise IKError("IK normal matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][size] for row in range(size)]


def _clamp_joints(
    joints: Sequence[float], bounds: Sequence[tuple[float, float]]
) -> list[float]:
    return [
        min(max(value, low), high)
        for value, (low, high) in zip(joints, bounds)
    ]


def _weighted_norm(
    error: Sequence[float], orientation_weight: float, active_axes: Sequence[int]
) -> float:
    weighted = [
        error[index] * (orientation_weight if index >= 3 else 1.0)
        for index in active_axes
    ]
    return math.sqrt(sum(value * value for value in weighted))


def solve_ik_dls(
    fk: Callable[[list[float]], list[float]],
    target_pose: Sequence[float],
    seed_joints: Sequence[float],
    joint_bounds: Sequence[tuple[float, float]],
    *,
    max_iterations: int = 80,
    position_tolerance: float = 0.0008,
    orientation_tolerance: float = math.radians(0.8),
    damping: float = 0.015,
    orientation_weight: float = 0.15,
    finite_difference: float = 1e-4,
    max_iteration_step: float = math.radians(5.0),
    active_axes: Sequence[int] | None = None,
    seed_weight: float = 0.0005,
    preferred_joints: Sequence[float] | None = None,
    preferred_weight: float = 0.0001,
    max_time_s: float | None = None,
) -> list[float]:
    """Solve IK near *seed_joints* using numerical spatial Jacobians."""
    if len(target_pose) != 6 or len(seed_joints) != 6 or len(joint_bounds) != 6:
        raise ValueError("Piper IK expects six pose, joint, and bound values")
    axes = tuple(range(6)) if active_axes is None else tuple(active_axes)
    if not axes or any(index not in range(6) for index in axes):
        raise ValueError("active_axes must contain pose indexes in 0..5")
    if preferred_joints is not None and len(preferred_joints) != 6:
        raise ValueError("preferred_joints must contain six values")
    joints = _clamp_joints(seed_joints, joint_bounds)
    deadline = None if max_time_s is None else time.perf_counter() + max_time_s
    orientation_active = any(index >= 3 for index in axes)

    def task_error(target, current):
        # Translation-only jogs do not need rotation matrices/logarithms for
        # each FK perturbation and line-search candidate.
        if not orientation_active:
            return [target[i] - current[i] for i in range(3)] + [0.0] * 3
        return pose_error(target, current)

    current_pose = fk(joints)

    for _ in range(max_iterations):
        if deadline is not None and time.perf_counter() >= deadline:
            raise IKError("IK time budget exceeded")
        error = task_error(target_pose, current_pose)
        position_error = math.sqrt(
            sum(error[index] ** 2 for index in axes if index < 3)
        )
        orientation_error = math.sqrt(
            sum(error[index] ** 2 for index in axes if index >= 3)
        )
        if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
            if orientation_active:
                return joints
        if position_error <= position_tolerance and not orientation_active:
            return joints

        jacobian = [[0.0] * 6 for _ in range(6)]
        for column in range(6):
            perturbed = joints[:]
            perturbed[column] += finite_difference
            perturbed_pose = fk(perturbed)
            delta = task_error(perturbed_pose, current_pose)
            for row in range(6):
                jacobian[row][column] = delta[row] / finite_difference

        weighted_error = [
            error[index] * (orientation_weight if index >= 3 else 1.0)
            for index in axes
        ]
        weighted_jacobian = [
            [
                jacobian[index][column]
                * (orientation_weight if index >= 3 else 1.0)
                for column in range(6)
            ]
            for index in axes
        ]

        # Joint-space DLS with seed/reference posture regularization.
        normal = [[0.0] * 6 for _ in range(6)]
        right_hand = [0.0] * 6
        for row in range(6):
            for column in range(6):
                normal[row][column] = sum(
                    jacobian_row[row] * jacobian_row[column]
                    for jacobian_row in weighted_jacobian
                )
            normal[row][row] += damping * damping + seed_weight
            right_hand[row] = sum(
                jacobian_row[row] * value
                for jacobian_row, value in zip(weighted_jacobian, weighted_error)
            ) + seed_weight * (seed_joints[row] - joints[row])
            if preferred_joints is not None and preferred_weight > 0:
                normal[row][row] += preferred_weight
                right_hand[row] += preferred_weight * (
                    preferred_joints[row] - joints[row]
                )
        delta_joints = _solve_linear(normal, right_hand)
        largest = max(abs(value) for value in delta_joints)
        if largest > max_iteration_step:
            scale = max_iteration_step / largest
            delta_joints = [value * scale for value in delta_joints]

        old_cost = _weighted_norm(error, orientation_weight, axes)
        accepted = False
        for line_scale in (1.0, 0.5, 0.25, 0.1):
            candidate = _clamp_joints(
                [
                    value + line_scale * delta
                    for value, delta in zip(joints, delta_joints)
                ],
                joint_bounds,
            )
            candidate_pose = fk(candidate)
            new_error = task_error(target_pose, candidate_pose)
            if _weighted_norm(new_error, orientation_weight, axes) < old_cost:
                joints = candidate
                current_pose = candidate_pose
                accepted = True
                break
        if not accepted:
            raise IKError("IK iteration stalled near a singularity or joint limit")

    final_error = task_error(target_pose, current_pose)
    position_mm = 1000.0 * math.sqrt(
        sum(final_error[index] ** 2 for index in axes if index < 3)
    )
    orientation_deg = math.degrees(
        math.sqrt(
            sum(final_error[index] ** 2 for index in axes if index >= 3)
        )
    )
    raise IKError(
        f"IK did not converge: position error={position_mm:.2f} mm, "
        f"orientation error={orientation_deg:.2f} deg"
    )
