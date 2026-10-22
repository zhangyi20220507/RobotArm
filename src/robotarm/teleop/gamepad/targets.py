"""Jog 目标构建与 DLS 求解：增量目标、关节目标累积、末端平滑。"""

from __future__ import annotations

import math
import time
from typing import Any

from ...core.kinematics.dls_ik import IKError, solve_ik_dls


def cartesian_pose(
    robot: Any, joints: list[float], tcp_offset: float = 0.0
) -> list[float]:
    """Return the controlled TCP pose for a local-Z offset from the flange."""
    pose = list(robot.fk(joints))
    if tcp_offset == 0.0:
        return pose
    roll, pitch, yaw = pose[3:6]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Third column of Rz(yaw) @ Ry(pitch) @ Rx(roll): flange local Z in base.
    pose[0] += tcp_offset * (cy * sp * cr + sy * sr)
    pose[1] += tcp_offset * (sy * sp * cr - cy * sr)
    pose[2] += tcp_offset * (cp * cr)
    return pose


def incremental_target(
    previous: float, feedback: float, direction: float, step: float, lead: float
) -> float:
    if (previous - feedback) * direction < 0:
        previous = feedback
    return min(max(previous + direction * step, feedback - lead), feedback + lead)


def clamp_orientation(axis: int, value: float) -> float:
    low, high = (-math.pi / 2, math.pi / 2) if axis == 4 else (-math.pi, math.pi)
    return min(max(value, low), high)


def solve_cartesian_target(
    robot: Any,
    joints: list[float],
    pose_target: list[float],
    axes: list[float],
    bounds: list[tuple[float, float]],
    position_steps: list[float],
    angle_step: float,
    max_position_lead: float,
    max_angle_lead: float,
    max_joint_command: float,
    preferred_joints: list[float],
    tcp_offset: float = 0.0,
    ik_solver: Any = None,
) -> tuple[list[float], list[float]]:
    def fk(values: list[float]) -> list[float]:
        return cartesian_pose(robot, values, tcp_offset)

    current_pose = fk(joints)
    # Keep uncommanded components in the persistent target buffer.  Position
    # jogs solve against the offset TCP below, so its inactive XYZ components
    # remain fixed even when the flange orientation changes.
    requested = pose_target[:]
    for axis in range(3):
        if axes[axis]:
            requested[axis] = incremental_target(
                pose_target[axis], current_pose[axis],
                1.0 if axes[axis] > 0 else -1.0,
                position_steps[axis] * abs(axes[axis]), max_position_lead,
            )
    for axis in range(3, 6):
        if axes[axis]:
            requested[axis] = clamp_orientation(
                axis,
                incremental_target(
                    pose_target[axis], current_pose[axis],
                    1.0 if axes[axis] > 0 else -1.0,
                    angle_step * abs(axes[axis]), max_angle_lead,
                ),
            )
    position_active = any(axes[index] for index in range(3))
    orientation_axes = tuple(i for i in range(3, 6) if axes[i])
    # Position jog constrains the actual gripper TCP rather than the flange.
    # Orientation remains free, preserving the redundant DOFs and reachability
    # of the original three-axis DLS behavior without letting the grasp point
    # sag when the wrist tilts.
    active_axes = (0, 1, 2) if position_active else (0, 1, 2, *orientation_axes)
    # DLS gets one shared wall-clock budget, including retries. Shrink only
    # user-commanded components; never relax the held axes or safety limits.
    deadline = time.perf_counter() + 0.015
    failures = []
    for scale in ((1.0,) if ik_solver is not None else (1.0, 0.5, 0.25)):
        candidate = requested[:]
        for axis, value in enumerate(axes):
            if value:
                candidate[axis] = current_pose[axis] + scale * (requested[axis] - current_pose[axis])
        options = {} if ik_solver is not None else {
            "max_time_s": max(0.0, deadline - time.perf_counter()),
            "max_iterations": 32,
        }
        try:
            solution = (ik_solver or solve_ik_dls)(
                fk, candidate, joints, bounds,
                active_axes=active_axes,
                orientation_tolerance=math.radians(0.05),
                seed_weight=0.0002,
                preferred_joints=preferred_joints,
                preferred_weight=0.00002,
                **options,
            )
            if not all(math.isfinite(q) and lo - 1e-6 <= q <= hi + 1e-6
                       for q, (lo, hi) in zip(solution, bounds)) or len(solution) != 6:
                raise IKError("IK returned invalid or out-of-bounds joints")
            largest = max(abs(target - current) for target, current in zip(solution, joints))
            if largest > max_joint_command:
                raise IKError(f"最大关节变化 {math.degrees(largest):.2f}° 超过命令阈值")
            if ik_solver is None and time.perf_counter() > deadline:
                raise IKError("IK time budget exceeded")
            return solution, candidate
        except IKError as exc:
            failures.append(f"{scale:g}x: {exc}")
            if ik_solver is not None or time.perf_counter() >= deadline:
                break
    raise IKError("; ".join(failures))


def build_joint_targets(
    previous_targets: list[float],
    current_joints: list[float],
    inputs: list[float],
    step: float,
    max_lead: float,
    bounds: list[tuple[float, float]],
) -> tuple[list[float], tuple[int, ...]]:
    """Accumulate active joint targets, bounded against fresh feedback."""
    targets = current_joints[:]
    active: list[int] = []
    for index, value in enumerate(inputs):
        if value == 0.0:
            continue
        active.append(index)
        direction = 1.0 if value > 0 else -1.0
        origin = previous_targets[index]
        if (origin - current_joints[index]) * direction < 0:
            origin = current_joints[index]
        candidate = origin + value * step
        candidate = min(
            max(candidate, current_joints[index] - max_lead),
            current_joints[index] + max_lead,
        )
        low, high = bounds[index]
        if current_joints[index] <= low and direction < 0:
            candidate = current_joints[index]
        elif current_joints[index] >= high and direction > 0:
            candidate = current_joints[index]
        else:
            candidate = min(max(candidate, low), high)
        targets[index] = candidate
    return targets, tuple(active)


def interpolate_cartesian_joints(
    previous_command: list[float],
    desired_solution: list[float],
    feedback: list[float],
    smoothing: float,
    max_lead: float,
    bounds: list[tuple[float, float]],
) -> list[float]:
    """Low-pass a DLS solution before MOVE_JS transmission, without speed limiting."""
    command: list[float] = []
    for previous, desired, current, (low, high) in zip(
        previous_command, desired_solution, feedback, bounds
    ):
        # Re-anchor stale command state to current feedback after mode/pause delays.
        previous = min(max(previous, current - max_lead), current + max_lead)
        candidate = previous + smoothing * (desired - previous)
        candidate = min(max(candidate, current - max_lead), current + max_lead)
        command.append(min(max(candidate, low), high))
    return command


def jog_step_levels(initial_step_deg: float, max_lead_deg: float) -> tuple[list[float], int]:
    """Build software Joint Jog increments; JS mode ignores speed_percent."""
    candidates = (
        initial_step_deg * 0.25,
        initial_step_deg * 0.5,
        initial_step_deg,
        initial_step_deg * 1.5,
        initial_step_deg * 2.0,
        initial_step_deg * 3.0,
        initial_step_deg * 5.0,
        initial_step_deg * 8.0,
    )
    levels = sorted(
        {
            round(max(0.02, min(value, 5.0, max_lead_deg)), 4)
            for value in candidates
        }
    )
    initial = round(min(initial_step_deg, 5.0, max_lead_deg), 4)
    if initial not in levels:
        levels.append(initial)
        levels.sort()
    return levels, levels.index(initial)
