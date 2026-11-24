#!/usr/bin/env python3
"""Piper policy runner; uncalibrated Pi0.5 is restricted to shadow inference."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

from ..config import defaults, policy_defaults
from ..core.camera.realsense import RoleCameras
from ..core.robot import piper as base
from ..paths import (
    ARTIFACTS_ROOT,
    MODELING_CONFIGS_ROOT,
    PIPER_RAW_ROOT,
    TRAINING_LOG_ROOT,
    WORKSPACE_ROOT,
)
from ..record.recorder import RawEpisodeRecorder
from ..record.schema import HG_DAGGER_SAMPLE_FIELDS
from ..teleop import keyboard as keyboard_api
from .hg_dagger import (
    LeaderExpert,
    LeaderHomingFailure,
    TakeoverMotionGate,
)
from .policy_recording import PolicyRecordingManager


ACT_STREAM_SPEED_PERCENT = 100  # Same controller setting as gamepad collection.
ACT_RESET_RELEASE_WIDTH_M = 0.08
ACT_RESET_GRIPPER_TIMEOUT_S = 8.0


def workspace_path(value: str) -> Path:
    """Resolve a YAML path default against the workspace root."""
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def checkpoint_digest(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pi0_checkpoint_digest(checkpoint, base_checkpoint):
    """Bind Pi0 approval to adapter, base, processors and normalization stats."""
    digest = hashlib.sha256()
    files = sorted(p for p in checkpoint.iterdir()
                   if p.suffix in {".json", ".safetensors"})
    files += [base_checkpoint / "config.json", base_checkpoint / "model.safetensors"]
    for index, path in enumerate(files):
        digest.update(f"{index}:{path.name}\0".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def pi0_expired_steps(requested_at, now, rate):
    """Ceil to the next future timestep; never replay past actions after inference."""
    return max(0, math.ceil(max(0., now - requested_at) * rate - 1e-6))


def trim_pi0_plan(actions, plan, now, rate, action_steps):
    """Trim elapsed prefix and return actual original chunk index (not queue index)."""
    skipped = pi0_expired_steps(plan["requested_at"], now, rate)
    return deque(actions[skipped:action_steps]), skipped


def runtime_artifact_paths(checkpoint: Path, digest: str) -> tuple[Path, Path, Path]:
    """Return stable shadow, execute-report, and approval paths for a checkpoint."""
    checkpoint = checkpoint.resolve()
    run_name = checkpoint.parents[2].name
    step_name = checkpoint.parent.name
    stem = f"{run_name}_{step_name}_{digest[:12]}"
    report_root = TRAINING_LOG_ROOT / "shadow"
    approval = ARTIFACTS_ROOT / "reports" / "runtime_approvals" / f"{stem}.json"
    return report_root / f"{stem}.jsonl", report_root / f"{stem}_execute.jsonl", approval


def find_approval(checkpoint: Path, digest: str, policy_type: str | None = None) -> Path:
    """Prefer generated approval, then reuse a matching legacy approval."""
    canonical = runtime_artifact_paths(checkpoint, digest)[2]
    if canonical.is_file():
        return canonical
    for path in sorted(MODELING_CONFIGS_ROOT.glob("*approval*.json")):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        valid_runtime_binding = (
            policy_type not in {"diffusion", "smolvla", "pi0"}
            or (
                content.get("policy_type") == policy_type
                and isinstance(content.get("runtime"), dict)
            )
        )
        if (
            content.get("approved") is True
            and content.get("checkpoint_sha256") == digest
            and valid_runtime_binding
        ):
            return path
    return canonical


def validate_joint_feedback(joints, robot_config, tolerance_deg=3.0):
    """Accept known small Piper zero offsets, but reject implausible feedback."""
    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
        raise RuntimeError("Piper returned invalid joint feedback")
    tolerance = math.radians(tolerance_deg)
    for index, value in enumerate(joints, start=1):
        low, high = robot_config["joint_limits"][f"joint{index}"]
        if not low - tolerance <= value <= high + tolerance:
            raise RuntimeError(
                f"J{index} feedback {math.degrees(value):.2f}° is outside "
                f"the expected range including ±{tolerance_deg:.1f}° tolerance"
            )


def deployment_joint_bounds(robot_config):
    """Return software command bounds with Piper J2/J3 zero-offset handling."""
    margin = math.radians(1.0)
    result = []
    for index in range(6):
        low, high = robot_config["joint_limits"][f"joint{index + 1}"]
        zero_offset = math.radians(3.0) if index in (1, 2) else 0.0
        result.append((low - zero_offset + margin, high + zero_offset - margin))
    return result


def within_joint_bounds(joints, bounds):
    return all(low <= value <= high for value, (low, high) in zip(joints, bounds))


def transition_is_safe(current, target, bounds):
    """Allow in-range motion, or hold/recovery when feedback starts outside."""
    for before, after, (low, high) in zip(current, target, bounds):
        if low <= before <= high:
            if not low <= after <= high:
                return False
        elif before < low:
            if not before <= after <= low:
                return False
        elif not high <= after <= before:
            return False
    return True


def inference_timing_messages(
    inference_ms: float,
    warning_ms: float,
    stop_ms: float,
) -> tuple[list[str], list[str]]:
    """Classify a synchronous inference overrun as a warning or hard stop."""
    detail = f"inference {inference_ms:.1f}ms"
    if inference_ms > stop_ms:
        return [], [f"{detail} > {stop_ms:.1f}ms"]
    if inference_ms > warning_ms:
        return [f"{detail} > warning threshold {warning_ms:.1f}ms"], []
    return [], []


def read_valid_joints(robot, robot_config, timeout):
    joints = [
        float(value) for value in base.wait_for_feedback(robot, timeout).msg
    ]
    validate_joint_feedback(joints, robot_config)
    return joints


def read_mode_feedback(robot):
    status = robot.get_arm_status()
    if status is None:
        return None
    return base.value_as_int(status.msg.mode_feedback)


def wait_mode_feedback(robot, expected, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        mode = read_mode_feedback(robot)
        if mode == expected:
            return
        time.sleep(0.02)
    raise RuntimeError(
        f"Piper did not enter JS mode: feedback={read_mode_feedback(robot)}, "
        f"expected={expected}"
    )


def prepare_js_control(
    robot, robot_config, gripper, gripper_width, gripper_force, timeout,
    expected_js_mode, stream_speed_percent=None,
):
    """Enter Piper JS mode without exposing a stale/default joint target."""
    joints = read_valid_joints(robot, robot_config, timeout)
    # Preload and hold measured feedback in the smoother J mode before enabling.
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(joints)
    base.enable_arm_and_gripper(
        robot, gripper, gripper_width, gripper_force, timeout
    )
    if stream_speed_percent is not None:
        # An enable broadcast can reset controller settings; restore the
        # collection speed only after that broadcast has completed.
        robot.set_speed_percent(stream_speed_percent)
    time.sleep(0.1)
    joints = read_valid_joints(robot, robot_config, timeout)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
    robot.move_js(joints)
    wait_mode_feedback(robot, expected_js_mode, timeout)
    return joints


def move_gripper_and_wait(
    gripper, target_width, force, tolerance, timeout, feedback_timeout,
    emergency_stop_requested=None, progress_callback=None,
):
    """Wait for an R-reset gripper release/close before moving to the next phase."""
    if not math.isfinite(target_width) or target_width < 0:
        raise RuntimeError("invalid gripper reset target")
    if emergency_stop_requested is not None and emergency_stop_requested():
        raise InterruptedError("electronic emergency stop requested during gripper reset")
    gripper.move_gripper_m(target_width, force)
    deadline = time.monotonic() + timeout
    measured = None
    while time.monotonic() < deadline:
        if emergency_stop_requested is not None and emergency_stop_requested():
            raise InterruptedError("electronic emergency stop requested during gripper reset")
        measured = base.wait_gripper_width(gripper, min(feedback_timeout, 0.2))
        if not math.isfinite(measured) or not -0.003 <= measured <= 0.153:
            raise RuntimeError("invalid gripper feedback during reset")
        if progress_callback is not None:
            progress_callback(max(measured, 0.0), target_width)
        if abs(measured - target_width) <= tolerance:
            return max(measured, 0.0)
        time.sleep(0.05)
    raise TimeoutError(
        f"gripper reset did not reach {target_width * 1000:.0f} mm "
        f"within {timeout:.1f}s (feedback={measured})"
    )


def clamp_step(
    predicted, current, max_joint_step, max_gripper_step, gripper_max,
    joint_bounds, previous_target, max_joint_lead, max_gripper_lead,
):
    import numpy as np

    predicted = np.asarray(predicted, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if predicted.shape != (7,) or not np.isfinite(predicted).all():
        raise RuntimeError("policy returned a non-finite or non-7D action")
    previous = (
        current.copy() if previous_target is None
        else np.asarray(previous_target, dtype=np.float64)
    )
    if previous.shape != (7,) or not np.isfinite(previous).all():
        raise RuntimeError("previous command target is invalid")
    joint_delta = np.clip(
        predicted[:6] - previous[:6], -max_joint_step, max_joint_step
    )
    candidate = previous[:6] + joint_delta
    candidate = np.clip(
        candidate, current[:6] - max_joint_lead,
        current[:6] + max_joint_lead,
    )
    joints = candidate.copy()
    for index, (low, high) in enumerate(joint_bounds):
        before = current[index]
        if before < low:
            # Do not move farther out; recover by at most one limited step.
            joints[index] = min(max(candidate[index], before), low)
        elif before > high:
            joints[index] = max(min(candidate[index], before), high)
        else:
            joints[index] = min(max(candidate[index], low), high)
    gripper = float(np.clip(
        predicted[6], previous[6] - max_gripper_step,
        previous[6] + max_gripper_step,
    ))
    gripper = float(np.clip(
        gripper, current[6] - max_gripper_lead,
        current[6] + max_gripper_lead,
    ))
    gripper = float(np.clip(gripper, 0.0, gripper_max))
    return np.asarray([*joints, gripper], dtype=np.float64)


def action_queue_depth(policy):
    """Return the policy action-queue depth when the policy exposes one."""
    # ACT owns a deque directly, while async policies expose named queues.
    queue = getattr(policy, "_action_queue", None)
    if queue is not None:
        return len(queue)
    queues = getattr(policy, "_queues", None)
    if not isinstance(queues, dict):
        return None
    queue = queues.get("action")
    return None if queue is None else len(queue)


def action_was_clamped(raw, safe, tolerance=1e-6):
    """Return whether any action dimension changed materially."""
    import numpy as np

    return bool(np.any(np.abs(np.asarray(raw) - np.asarray(safe)) > tolerance))


def act_backpressure_required(shaped, safe, current):
    """Wait for joint/opening limits, but allow a force-limited grasp to hold.

    A closing gripper may never reach its requested width once an object is
    grasped. Its unchanged step/lead/force limits still constrain commands;
    width convergence must not freeze ACT's subsequent lifting actions.
    This is a direction rule, not a claim that contact was detected.
    """
    import numpy as np

    shaped, safe, current = (
        np.asarray(value, dtype=np.float64) for value in (shaped, safe, current)
    )
    if any(value.shape != (7,) or not np.isfinite(value).all()
           for value in (shaped, safe, current)):
        raise RuntimeError("ACT backpressure received an invalid 7D action/state")
    joint_limited = action_was_clamped(shaped[:6], safe[:6])
    gripper_limited = action_was_clamped(shaped[6:], safe[6:])
    closing = shaped[6] < current[6] and safe[6] <= current[6]
    return joint_limited or (gripper_limited and not closing)


def limit_joint_target_acceleration(
    desired, previous_target, previous_velocity, max_acceleration, dt,
    max_velocity,
):
    """Shape a 7D target with a trapezoidal joint velocity profile.

    Only the six arm joints are shaped; the gripper keeps its independent
    position/rate limiter. Inputs and outputs use radians and seconds.
    """
    import numpy as np

    desired = np.asarray(desired, dtype=np.float64)
    previous = np.asarray(previous_target, dtype=np.float64)
    velocity = np.asarray(previous_velocity, dtype=np.float64)
    if desired.shape != (7,) or previous.shape != (7,) or velocity.shape != (6,):
        raise RuntimeError("joint acceleration limiter received an invalid shape")
    if not (
        np.isfinite(desired).all()
        and np.isfinite(previous).all()
        and np.isfinite(velocity).all()
    ):
        raise RuntimeError("joint acceleration limiter received a non-finite value")
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("joint acceleration limiter dt must be positive")
    if max_acceleration <= 0:
        return desired.copy(), (desired[:6] - previous[:6]) / dt
    if max_velocity <= 0 or not math.isfinite(max_velocity):
        raise ValueError("joint acceleration limiter max velocity must be positive")

    displacement = desired[:6] - previous[:6]
    direction = np.sign(displacement)
    # Slow down early enough to stop at the requested position. The ordinary
    # step limiter remains the final hard velocity/position safety boundary.
    braking_speed = np.sqrt(2.0 * max_acceleration * np.abs(displacement))
    requested_velocity = direction * np.minimum(max_velocity, braking_speed)
    velocity_delta = max_acceleration * dt
    next_velocity = np.clip(
        requested_velocity, velocity - velocity_delta, velocity + velocity_delta
    )
    # If the exact one-tick landing velocity is reachable without breaking the
    # acceleration/velocity limits, use it and settle exactly at the target.
    landing_velocity = displacement / dt
    can_land = (
        np.abs(landing_velocity - velocity) <= velocity_delta
    ) & (np.abs(landing_velocity) <= max_velocity)
    next_velocity[can_land] = landing_velocity[can_land]
    shaped = desired.copy()
    shaped[:6] = previous[:6] + next_velocity * dt
    return shaped, next_velocity


def act_subgoal_errors(target, current):
    """Return joint L2/max error in radians and gripper error in metres."""
    import numpy as np

    target = np.asarray(target, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if target.shape != (7,) or current.shape != (7,):
        raise RuntimeError("ACT subgoal/action state shape is not 7D")
    if not np.isfinite(target).all() or not np.isfinite(current).all():
        raise RuntimeError("ACT subgoal/action state contains a non-finite value")
    joint_errors = np.abs(target[:6] - current[:6])
    return (
        float(np.linalg.norm(joint_errors)),
        float(np.max(joint_errors)),
        abs(float(target[6] - current[6])),
    )


def update_act_backpressure_progress(
    target, current, previous_joint_error, previous_gripper_error,
    no_progress_steps, min_joint_progress, min_gripper_progress,
):
    """Measure feedback progress while an ACT target remains hard-clamped."""
    if no_progress_steps < 0:
        raise ValueError("ACT subgoal no-progress steps cannot be negative")
    for name, value in (
        ("min_joint_progress", min_joint_progress),
        ("min_gripper_progress", min_gripper_progress),
    ):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")

    joint_error, max_joint_error, gripper_error = act_subgoal_errors(
        target, current
    )
    joint_progress = (
        None if previous_joint_error is None
        else previous_joint_error - joint_error
    )
    gripper_progress = (
        None if previous_gripper_error is None
        else previous_gripper_error - gripper_error
    )
    progressed = (
        previous_joint_error is None
        or previous_gripper_error is None
        or joint_progress >= min_joint_progress
        or gripper_progress >= min_gripper_progress
    )
    next_no_progress_steps = 0 if progressed else no_progress_steps + 1
    return {
        "joint_error": joint_error,
        "max_joint_error": max_joint_error,
        "gripper_error": gripper_error,
        "joint_progress": joint_progress,
        "gripper_progress": gripper_progress,
        "progressed": progressed,
        "no_progress_steps": next_no_progress_steps,
    }


def latch_act_subgoal(current_subgoal, safe_target, should_latch):
    """Keep an existing ACT subgoal fixed or latch a new safe target."""
    import numpy as np

    if current_subgoal is not None:
        subgoal = np.asarray(current_subgoal, dtype=np.float64)
        if subgoal.shape != (7,) or not np.isfinite(subgoal).all():
            raise RuntimeError("ACT latched subgoal is invalid")
        return subgoal, False
    if not should_latch:
        return None, False
    subgoal = np.asarray(safe_target, dtype=np.float64)
    if subgoal.shape != (7,) or not np.isfinite(subgoal).all():
        raise RuntimeError("ACT safe subgoal is invalid")
    return subgoal.copy(), True


def update_act_backpressure_target(
    current_subgoal, candidate_subgoal, tracking_clamped, raw_action_ok,
):
    """Latch while the joint/opening limiter is applying backpressure.

    Releasing on ``not tracking_clamped`` deliberately does not require the
    robot to settle at an exact point.  Piper JS can retain a small steady
    position residual even though the requested action is safe to advance.
    """
    if current_subgoal is not None:
        if not raw_action_ok or not tracking_clamped:
            return None, False, True
        retained, _ = latch_act_subgoal(current_subgoal, candidate_subgoal, True)
        return retained, False, False
    if tracking_clamped and raw_action_ok:
        created, _ = latch_act_subgoal(None, candidate_subgoal, True)
        return created, True, False
    return None, False, False


def bounded_act_subgoal(
    desired, current, joint_bounds, joint_velocity, max_joint_lead,
    max_gripper_lead, gripper_max, max_joint_acceleration,
):
    """Build one fixed local target with enough room for smooth braking.

    The command itself remains bounded by ``max_joint_lead`` every tick.  The
    fixed subgoal may be farther away by the current velocity's braking
    distance so the acceleration limiter can stop at it without continually
    moving the endpoint forward.
    """
    import numpy as np

    desired = np.asarray(desired, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    if desired.shape != (7,) or current.shape != (7,) or velocity.shape != (6,):
        raise RuntimeError("ACT bounded subgoal received an invalid shape")
    if len(joint_bounds) != 6:
        raise RuntimeError("ACT bounded subgoal requires six joint bounds")
    if not (
        np.isfinite(desired).all()
        and np.isfinite(current).all()
        and np.isfinite(velocity).all()
    ):
        raise RuntimeError("ACT bounded subgoal received a non-finite value")
    if max_joint_lead <= 0 or max_gripper_lead <= 0 or gripper_max <= 0:
        raise ValueError("ACT bounded subgoal limits must be positive")
    if max_joint_acceleration < 0 or not math.isfinite(max_joint_acceleration):
        raise ValueError("ACT bounded subgoal acceleration must be non-negative")

    braking_distance = (
        np.square(velocity) / (2.0 * max_joint_acceleration)
        if max_joint_acceleration > 0
        else np.zeros(6, dtype=np.float64)
    )
    local_reach = max_joint_lead + braking_distance
    joints = np.clip(
        desired[:6], current[:6] - local_reach, current[:6] + local_reach
    )
    joints = np.asarray([
        np.clip(value, low, high)
        for value, (low, high) in zip(joints, joint_bounds)
    ], dtype=np.float64)
    gripper = float(np.clip(
        desired[6],
        max(0.0, current[6] - max_gripper_lead),
        min(gripper_max, current[6] + max_gripper_lead),
    ))
    return np.asarray([*joints, gripper], dtype=np.float64)


def act_tracking_action(raw, gripper_max, gripper_bound_tolerance):
    """Use legal gripper endpoints for tolerated prediction overshoot.

    Preserve raw output for envelope checks and telemetry. Overshoot beyond
    tolerance stays untouched here so it is still rejected by the raw gate.
    """
    import numpy as np

    if gripper_bound_tolerance < 0 or not math.isfinite(gripper_bound_tolerance):
        raise ValueError("gripper bound tolerance must be finite and non-negative")
    action = np.asarray(raw, dtype=np.float64).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError("policy returned a non-finite or non-7D action")
    if -gripper_bound_tolerance <= action[6] <= gripper_max + gripper_bound_tolerance:
        action[6] = np.clip(action[6], 0.0, gripper_max)
    return action


def act_action_within_envelope(
    raw, current, joint_bounds, gripper_max,
    max_raw_joint_gap, max_raw_gripper_gap,
    raw_gripper_bound_tolerance=0.0,
    raw_joint_bound_tolerance=0.0,
):
    """Reject ACT targets outside hard bounds or the demonstrated gap envelope.

    The gap limit is deliberately applied to the raw policy output, before the
    rate limiter. Otherwise a bad absolute target could be made superficially
    safe one tick at a time and eventually be followed all the way there.
    """
    return not act_action_envelope_failures(
        raw, current, joint_bounds, gripper_max,
        max_raw_joint_gap, max_raw_gripper_gap,
        raw_gripper_bound_tolerance,
        raw_joint_bound_tolerance,
    )


def act_joint_gap_failure(raw, current, max_raw_joint_gap):
    """Return the largest raw joint-gap violation, if the inputs are usable."""
    import numpy as np

    raw = np.asarray(raw, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if (
        raw.shape != (7,)
        or current.shape != (7,)
        or not np.isfinite(raw).all()
        or not np.isfinite(current).all()
    ):
        return None
    joint_gaps = np.abs(raw[:6] - current[:6])
    largest_joint = int(np.argmax(joint_gaps))
    if joint_gaps[largest_joint] <= max_raw_joint_gap:
        return None
    return (
        f"raw joint gap J{largest_joint + 1}="
        f"{math.degrees(joint_gaps[largest_joint]):.2f}deg > "
        f"{math.degrees(max_raw_joint_gap):.2f}deg"
    )


def update_act_raw_gap_streak(failure, steps, max_steps):
    """Count a severe ACT raw joint gap for warning telemetry."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    next_steps = steps + 1 if failure is not None else 0
    persistent_failure = None
    if failure is not None and next_steps >= max_steps:
        persistent_failure = (
            failure + f"; persisted for {next_steps} consecutive ticks"
        )
    return next_steps, persistent_failure


def act_action_envelope_failures(
    raw, current, joint_bounds, gripper_max,
    max_raw_joint_gap, max_raw_gripper_gap,
    raw_gripper_bound_tolerance=0.0,
    raw_joint_bound_tolerance=0.0,
):
    """Return human-readable reasons an ACT target cannot be executed."""
    import numpy as np

    raw = np.asarray(raw, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if raw.shape != (7,) or current.shape != (7,):
        return ["ACT action/state shape is not 7D"]
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        return ["ACT action/state contains a non-finite value"]

    reasons = []
    outside = []
    if (
        raw_joint_bound_tolerance < 0.0
        or not math.isfinite(raw_joint_bound_tolerance)
    ):
        raise ValueError(
            "raw joint bound tolerance must be finite and non-negative"
        )
    for index, (value, (low, high)) in enumerate(
        zip(raw[:6], joint_bounds), start=1
    ):
        if not (
            low - raw_joint_bound_tolerance
            <= value
            <= high + raw_joint_bound_tolerance
        ):
            outside.append(
                f"J{index}={math.degrees(value):.2f}deg "
                f"outside [{math.degrees(low):.2f}, {math.degrees(high):.2f}]deg "
                f"beyond {math.degrees(raw_joint_bound_tolerance):.2f}deg tolerance"
            )
    if outside:
        reasons.append("raw joint bound: " + ", ".join(outside))
    if (
        raw_gripper_bound_tolerance < 0.0
        or not math.isfinite(raw_gripper_bound_tolerance)
    ):
        raise ValueError("raw gripper bound tolerance must be finite and non-negative")
    gripper_low = -raw_gripper_bound_tolerance
    gripper_high = gripper_max + raw_gripper_bound_tolerance
    if not gripper_low <= raw[6] <= gripper_high:
        reasons.append(
            f"raw gripper={raw[6] * 1000.0:.2f}mm outside "
            f"[{gripper_low * 1000.0:.2f}, "
            f"{gripper_high * 1000.0:.2f}]mm"
        )

    joint_gap_failure = act_joint_gap_failure(
        raw, current, max_raw_joint_gap
    )
    if joint_gap_failure is not None:
        reasons.append(joint_gap_failure)
    gripper_gap = abs(float(raw[6] - current[6]))
    if gripper_gap > max_raw_gripper_gap:
        reasons.append(
            f"raw gripper gap={gripper_gap * 1000.0:.2f}mm > "
            f"{max_raw_gripper_gap * 1000.0:.2f}mm"
        )
    return reasons


def add_smolvla_task(batch, policy_type, task):
    """Attach the language instruction required by SmolVLA preprocessing."""
    if policy_type in {"smolvla", "pi0"}:
        batch["task"] = [task]
    return batch


class AsyncPlanner:
    """Generate complete action chunks off the control thread.

    Used by async-planned policies (SmolVLA, Diffusion, Pi0 LoRA). The policy (and
    its private action queue) is owned exclusively by this worker.  The
    control thread only consumes immutable NumPy actions, so a CUDA
    inference can never delay a Piper control tick.
    """

    def __init__(self, policy, preprocessor, postprocessor, reset, action_steps,
                 full_chunk=False, amp=False):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.reset = reset
        self.action_steps = action_steps
        self.full_chunk = full_chunk
        self.amp = amp
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._failure = None
        self._next_request_id = 0
        self._thread = threading.Thread(
            target=self._run, name="policy-planner", daemon=True
        )

    def start(self):
        self._thread.start()

    def request(self, batch, requested_at):
        """Request one future chunk using this immutable observation snapshot."""
        self._next_request_id += 1
        request_id = self._next_request_id
        item = (request_id, batch, requested_at)
        try:
            self._requests.put_nowait(item)
        except queue.Full:
            # A newer snapshot is more useful than a request that has not begun.
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            self._requests.put_nowait(item)
        return request_id

    def take_ready(self, minimum_request_id=0):
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return None
            if result["request_id"] >= minimum_request_id:
                return result

    def wait_ready(self, timeout, minimum_request_id=0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self._results.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                return None
            if result["request_id"] >= minimum_request_id:
                return result
        return None

    @property
    def failure(self):
        return self._failure

    def close(self):
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self):
        import torch

        while not self._stop.is_set():
            request = self._requests.get()
            if request is None:
                return
            request_id, batch, requested_at = request
            try:
                # Never share the policy's private queue with the control loop.
                if callable(self.reset):
                    self.reset()
                started = time.monotonic()
                actions = []
                with torch.inference_mode():
                    if self.full_chunk:
                        prepared = self.preprocessor(batch)
                        from contextlib import nullcontext
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()
                        with context:
                            chunk = self.policy.predict_action_chunk(prepared)
                        chunk = self.postprocessor(chunk).detach().float().cpu().numpy()
                        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 7:
                            raise RuntimeError("Pi0 planner requires [1,chunk,7] output")
                        actions = [value.copy() for value in chunk[0]]
                        import numpy as np
                        if not np.isfinite(chunk).all():
                            raise RuntimeError("Pi0 planner returned nonfinite actions")
                    for _ in range(0 if self.full_chunk else self.action_steps):
                        predicted = self.postprocessor(
                            self.policy.select_action(self.preprocessor(batch))
                        )
                        actions.append(
                            predicted.detach().cpu().squeeze(0).numpy().copy()
                        )
                completed_at = time.monotonic()
                result = {
                    "request_id": request_id,
                    "actions": tuple(actions),
                    "requested_at": requested_at,
                    "completed_at": completed_at,
                    "inference_ms": (completed_at - started) * 1000.0,
                }
                # Keep exactly one complete future chunk. A stale unconsumed
                # chunk is less safe than the newest completed plan.
                try:
                    self._results.put_nowait(result)
                except queue.Full:
                    try:
                        self._results.get_nowait()
                    except queue.Empty:
                        pass
                    self._results.put_nowait(result)
            except Exception as error:  # surfaced to the watchdog thread
                self._failure = f"{type(error).__name__}: {error}"
                return


class KeyboardIntentTracker:
    """Convert keyboard events into policy and recording intents."""

    def __init__(self, hold_seconds, repeat_gap=0.75):
        self.hold_seconds = hold_seconds
        self.repeat_gap = repeat_gap
        self.last_x = None
        self.last_r = None
        self.last_v = None
        self.last_i = None
        self.last_d = None
        self.y_started = None
        self.last_y = None
        self.y_event_count = 0
        self.y_fired = False

    def update(self, events, now):
        x_toggle = False
        reset_home = False
        recording_toggle = False
        emergency_stop = False
        for key, observed_at in events:
            if key == "x":
                if (
                    self.last_x is None
                    or observed_at - self.last_x > self.repeat_gap
                ):
                    x_toggle = True
                self.last_x = observed_at
            elif key == "r":
                if (
                    self.last_r is None
                    or observed_at - self.last_r > self.repeat_gap
                ):
                    reset_home = True
                self.last_r = observed_at
            elif key == "v":
                if (
                    self.last_v is None
                    or observed_at - self.last_v > self.repeat_gap
                ):
                    recording_toggle = True
                self.last_v = observed_at
            elif key == "y":
                if (
                    self.last_y is None
                    or observed_at - self.last_y > self.repeat_gap
                ):
                    self.y_started = observed_at
                    self.y_event_count = 1
                    self.y_fired = False
                else:
                    self.y_event_count += 1
                self.last_y = observed_at
            elif key == "alt+s":
                emergency_stop = True
        y_hold = (
            not self.y_fired
            and self.y_started is not None
            and self.last_y is not None
            and self.y_event_count >= 2
            and self.last_y - self.y_started >= self.hold_seconds
            and now - self.last_y <= self.repeat_gap
        )
        if y_hold:
            self.y_fired = True
        return x_toggle, y_hold, reset_home, emergency_stop, recording_toggle

    def intervention_toggle(self, events):
        """Debounce the HG-DAgger I gate without changing update()'s API."""
        toggle = False
        for key, observed_at in events:
            if key != "i":
                continue
            if self.last_i is None or observed_at - self.last_i > 0.25:
                toggle = True
            self.last_i = observed_at
        return toggle

    def trial_discard(self, events):
        """Debounce the HG-DAgger D discard action."""
        discard = False
        for key, observed_at in events:
            if key != "d":
                continue
            if self.last_d is None or observed_at - self.last_d > self.repeat_gap:
                discard = True
            self.last_d = observed_at
        return discard

    def after_home(self, now):
        """Require a fresh X press; suppress R key repeats after homing."""
        self.last_x = None
        self.last_r = now
        self.last_i = None
        self.last_d = None
        self.y_started = None
        self.last_y = None
        self.y_event_count = 0
        self.y_fired = False


def home_arm_to_zero(
    robot, robot_config, gripper, gripper_width, gripper_force,
    feedback_timeout, speed_percent, timeout, emergency_stop_requested=None,
    progress_callback=None, companion=None,
):
    """Move all six joints to zero, then close the gripper with verified feedback."""
    def check_emergency_stop():
        if emergency_stop_requested is not None and emergency_stop_requested():
            if companion is not None:
                companion.emergency_stop()
            raise InterruptedError("electronic emergency stop requested during homing")

    check_emergency_stop()
    target = [0.0] * 6
    bounds = deployment_joint_bounds(robot_config)
    if not within_joint_bounds(target, bounds):
        raise RuntimeError("zero joint target is outside deployment bounds")

    joints = read_valid_joints(robot, robot_config, feedback_timeout)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(joints)
    base.enable_arm_and_gripper(
        robot, gripper, gripper_width, gripper_force, feedback_timeout
    )
    check_emergency_stop()
    # The all-joint enable broadcast is a controller state change that can
    # discard a previously sent motion mode. Speed and J mode must be
    # re-asserted AFTER enabling (same order as the SDK demo and as
    # prepare_js_control, which re-sets its mode after the enable call).
    # With the mode set only before the enable, the controller drops the
    # MOVE_J target and homing times out with the arm motionless.
    robot.set_speed_percent(speed_percent)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    time.sleep(0.1)
    joints = read_valid_joints(robot, robot_config, feedback_timeout)
    robot.move_j(joints)
    gripper.move_gripper_m(gripper_width, gripper_force)
    check_emergency_stop()
    leader_stop = threading.Event()
    leader_home_target = [*target, gripper_width]
    def leader_stop_requested():
        return leader_stop.is_set()

    def run_leader_home():
        companion.start_homing(
            leader_home_target, speed_percent,
            timeout=timeout, stop_requested=leader_stop_requested,
        )

    # Start the leader's verified firmware home on its own CAN channel, then
    # immediately send the follower's zero target. Both motions can overlap.
    # Claim leader ownership BEFORE scheduling the worker: it may not run
    # before the follower's first progress callback fires.
    if companion is not None:
        companion.homing_active = True
    with ThreadPoolExecutor(max_workers=1) as home_executor:
        leader_home = (
            home_executor.submit(run_leader_home)
            if companion is not None else None
        )
        try:
            # move_j(joints) above consumes the planned-J trigger. Re-arm it.
            robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
            robot.move_j(target)

            started = time.monotonic()
            deadline = started + timeout
            tolerance = math.radians(3.0)
            first_error = None
            next_home_status_at = started
            follower_gripper_closed = False
            no_motion_deadline = None
            print(
                (
                    "Homing follower: "
                    if companion is not None
                    else "Homing: "
                )
                + f"MOVE_J to zero at {speed_percent}% speed; "
                "gripper holds its width during travel, then closes at home.",
                flush=True,
            )
            while time.monotonic() < deadline:
                check_emergency_stop()
                joints = read_valid_joints(
                    robot, robot_config, min(feedback_timeout, 0.2)
                )
                status = robot.get_arm_status()
                max_error = max(abs(value) for value in joints)
                if progress_callback is not None:
                    progress_callback(
                        [*joints, gripper_width],
                        [*target, gripper_width],
                        "homing",
                    )
                if status is not None:
                    code = base.value_as_int(status.msg.arm_status)
                    err = getattr(status.msg, "err_status", None)
                    limit_hits = [
                        index + 1 for index in range(6)
                        if getattr(err, f"joint_{index + 1}_angle_limit", False)
                    ]
                    if code != 0 or limit_hits:
                        raise RuntimeError(
                            "Piper rejected the homing motion: "
                            f"arm_status={code}"
                            + (
                                f", angle-limit flags on joints {limit_hits}"
                                if limit_hits else ""
                            )
                        )
                if first_error is None:
                    first_error = max_error
                    no_motion_deadline = time.monotonic() + 3.0
                motion_done = (
                    status is not None
                    and base.value_as_int(status.msg.motion_status) == 0
                )
                if leader_home is not None and leader_home.done():
                    leader_home.result()
                companion_done = (
                    companion is None or (
                        leader_home.done() and companion.homing_ready()
                    )
                )
                companion_status = ""
                leader_error = (
                    None
                    if companion is None
                    else getattr(companion, "homing_error_degrees", None)
                )
                if isinstance(leader_error, (int, float)) and math.isfinite(leader_error):
                    companion_status = (
                        f" leader_error={leader_error:6.2f} deg"
                    )
                if time.monotonic() >= next_home_status_at:
                    print(
                        f"\rHOMING follower_error={math.degrees(max_error):6.2f} deg"
                        + companion_status,
                        end="", flush=True,
                    )
                    next_home_status_at = time.monotonic() + 1.0
                follower_ready = (
                    motion_done
                    and time.monotonic() - started >= 0.5
                    and max_error <= tolerance
                )
                if follower_ready and not follower_gripper_closed:
                    robot.move_j(joints)
                    print("\nFollower is at zero; closing its gripper now.", flush=True)
                    closed_width = move_gripper_and_wait(
                        gripper, 0.0, gripper_force, 0.003,
                        ACT_RESET_GRIPPER_TIMEOUT_S, feedback_timeout,
                        emergency_stop_requested,
                        (
                            None
                            if progress_callback is None
                            else lambda measured, commanded: progress_callback(
                                [*joints, measured],
                                [*joints, commanded],
                                "home_gripper_close",
                            )
                        ),
                    )
                    gripper_width = closed_width
                    follower_gripper_closed = True
                if follower_ready and follower_gripper_closed and companion_done:
                    if companion is not None:
                        companion.finish_homing([*joints, gripper_width])
                    print(
                        "Homing complete; both arms are at zero and the "
                        "follower gripper is closed."
                        if companion is not None
                        else "Homing complete; arm and gripper are at zero.",
                        flush=True,
                    )
                    return joints
                if (
                    time.monotonic() >= no_motion_deadline
                    and first_error > tolerance
                    and max_error > first_error - math.radians(0.5)
                ):
                    raise RuntimeError(
                        "homing command produced no joint motion within 3s "
                        "(joint_error stuck at "
                        f"{math.degrees(max_error):.2f} deg); the controller likely "
                        "dropped the J-mode MOVE_J target"
                    )
                if leader_home is not None and not leader_home.done():
                    leader_stop.wait(0.05)
                else:
                    time.sleep(0.05)
            raise TimeoutError(
                f"homing did not complete within {timeout:.1f} seconds "
                f"(joint_error={math.degrees(max_error):.2f} deg)"
            )

        finally:
            leader_stop.set()


def hold_follower_after_leader_home_failure(
    robot, robot_config, feedback_timeout,
):
    """Cancel the follower's home target without electronically stopping it."""
    joints = read_valid_joints(
        robot, robot_config, min(float(feedback_timeout), 0.5)
    )
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(joints)
    return joints


def isolate_leader_home_failure(
    robot, robot_config, feedback_timeout, companion,
):
    """Stop only the failed leader; keep the follower holding if possible."""
    follower_hold_error = None
    leader_stop_error = None
    try:
        hold_follower_after_leader_home_failure(
            robot, robot_config, feedback_timeout
        )
    except Exception as exc:
        follower_hold_error = exc
        robot.electronic_emergency_stop()
    try:
        companion.emergency_stop()
    except Exception as exc:
        leader_stop_error = exc
    return follower_hold_error, leader_stop_error


def engage_expert_with_follower_hold(robot, gripper, expert, state, joint_bounds,
                                    gripper_force, gripper_max):
    """Cancel the follower's prior target before a blocking leader-role change."""
    measured = [float(value) for value in state]
    if len(measured) != 7 or not all(math.isfinite(value) for value in measured):
        raise ValueError("takeover requires a finite 7D follower state")
    robot.move_js(measured[:6])
    gripper.move_gripper_m(min(max(measured[6], 0.0), gripper_max), gripper_force)
    return expert.engage(measured, joint_bounds)


def reset_act_to_home(
    robot, robot_config, gripper, args, expected_js_mode,
    emergency_stop_requested, progress_callback=None, companion=None,
):
    """Release any object at the current pose, home, and restore ACT's start grip."""
    if emergency_stop_requested():
        raise InterruptedError("electronic emergency stop requested during ACT reset")
    joints = read_valid_joints(robot, robot_config, args.feedback_timeout)
    robot.move_js(joints)  # Hold the present arm pose while releasing the object.
    release_width = min(args.gripper_max_mm / 1000.0, ACT_RESET_RELEASE_WIDTH_M)
    print(f"R reset: holding arm; opening gripper to {release_width * 1000:.0f} mm before homing.", flush=True)
    open_width = move_gripper_and_wait(
        gripper, release_width, args.gripper_force, 0.005,
        ACT_RESET_GRIPPER_TIMEOUT_S, args.feedback_timeout,
        emergency_stop_requested,
        (
            None
            if progress_callback is None
            else lambda measured, commanded: progress_callback(
                [*joints, measured],
                [*joints, commanded],
                "reset_gripper_open",
            )
        ),
    )
    home_arm_to_zero(
        robot, robot_config, gripper, open_width, args.gripper_force,
        args.feedback_timeout, args.home_speed_percent, args.home_timeout,
        emergency_stop_requested,
        progress_callback,
        companion,
    )
    closed_width = max(0.0, base.wait_gripper_width(gripper, args.feedback_timeout))
    prepare_js_control(
        robot, robot_config, gripper, closed_width, args.gripper_force,
        args.feedback_timeout, expected_js_mode, ACT_STREAM_SPEED_PERCENT,
    )
    return closed_width


def policy_overrides(policy_type: str) -> dict:
    """按策略类型叠加 YAML per-policy 默认值（显式 CLI 参数优先级最高）。

    policy_type 在加载 checkpoint 后才确定，而推理时限类参数
    （max-inference-ms 等）在不同策略间差异大，因此在 parse_args 之后叠加。
    只返回与平铺默认不同的键（per-policy 差异），避免重复设置已应用的值。
    chunk_size 等模型结构参数以 checkpoint 为准；n_action_steps 和 temporal
    ensembling 可由策略默认值调整在线执行节奏，显式 CLI 参数优先。
    """
    flat = defaults("modeling", "online-policy")
    merged = policy_defaults("modeling", "online-policy", policy_type)
    applied = {}
    for key, value in merged.items():
        if key in flat and flat[key] == value:
            continue
        flag = "--" + key.replace("_", "-")
        if any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
            continue
        applied[key] = value
    return applied


def approval_required() -> bool:
    """Return whether online execution requires a shadow approval."""
    required = defaults("modeling", "online-policy").get("approval-required")
    if not isinstance(required, bool):
        raise RuntimeError(
            "online-policy.approval-required must be a boolean in modeling.yaml"
        )
    return required


def validate_execution_approval(
    path: Path, digest: str, policy_type: str, runtime: dict
) -> bool:
    """Validate execution approval when required; return whether it was checked."""
    # Pi0 shadow remains available for latency/action analysis, but it is not
    # a prerequisite for execution.  Motion still requires the explicit Pi0
    # experimental switch and all runtime watchdogs remain active.
    if policy_type == "pi0":
        return False
    if not approval_required():
        return False
    if not path.is_file():
        raise RuntimeError(
            "execution locked: no approval for this checkpoint; run "
            f"`robotarm modeling validate-runtime -- --checkpoint {path}`"
        )
    validate_approval(path, digest, policy_type, runtime)
    return True


def required_camera_roles(input_features) -> list[str]:
    """Map checkpoint image input keys to the camera roles this runner can provide."""
    image_keys = {key for key in input_features if key.startswith("observation.images.")}
    role_to_key = {
        "global": "observation.images.global",
        "wrist": "observation.images.wrist",
    }
    supported = set(role_to_key.values())
    if "observation.state" not in input_features or not image_keys:
        raise RuntimeError(
            "online-policy requires observation.state and at least one observation.images.* input"
        )
    if not image_keys <= supported:
        raise RuntimeError(
            "online-policy cannot provide checkpoint camera inputs: "
            f"{sorted(image_keys - supported)}; supported inputs are global and wrist"
        )
    if "observation.images.global" not in image_keys:
        raise RuntimeError("online-policy requires the global camera input")
    extra = set(input_features) - {"observation.state"} - image_keys
    if extra:
        raise RuntimeError(f"online-policy cannot provide checkpoint inputs {sorted(extra)}")
    return [role for role in role_to_key if role_to_key[role] in image_keys]


def camera_batch_images(frames, camera_roles, cv2, torch):
    """Convert each required BGR camera frame to a normalized batch tensor."""
    batch = {}
    ages_ms = {}
    now = time.monotonic()
    for role in camera_roles:
        if role not in frames:
            raise RuntimeError(f"required {role} camera frame is unavailable")
        image_bgr, captured_at, _, _ = frames[role]
        ages_ms[role] = max(0.0, (now - captured_at) * 1000.0)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch[f"observation.images.{role}"] = torch.from_numpy(image_rgb).permute(
            2, 0, 1
        ).float().div(255).unsqueeze(0)
    return batch, ages_ms


def parse_args(argv: list[str] | None = None):
    cfg = defaults("modeling", "online-policy")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None,
                        help="Pi0 LoRA base weights; default artifacts/pretrained/pi0_base")
    parser.add_argument("--experimental-pi0-execute", action="store_true",
                        help="explicitly enable the gated Pi0 LoRA experimental motion path")
    parser.add_argument("--camera-serial", default=cfg["camera-serial"],
                        help="global/overhead camera RealSense serial")
    parser.add_argument("--wrist-camera-serial", default=cfg["wrist-camera-serial"],
                        help="wrist camera RealSense serial (used when checkpoint needs it)")
    parser.add_argument("--camera-fps", type=int, default=cfg["camera-fps"])
    parser.add_argument(
        "--recording-root",
        type=Path,
        default=ARTIFACTS_ROOT / "policy_recordings",
        help=(
            "V-key recording output root; the session directory is created "
            "only when recording starts"
        ),
    )
    parser.add_argument(
        "--recording-fps",
        type=int,
        default=None,
        help="recording FPS; defaults to min(control rate, camera FPS)",
    )
    parser.add_argument(
        "--recording-encoder-queue-size",
        type=int,
        default=120,
        help="queued frames per asynchronous video encoder",
    )
    parser.add_argument(
        "--hg-dagger",
        action="store_true",
        help=(
            "enable human-gated DAgger: I takes over with a Piper leader arm, "
            "I again saves the expert recovery and returns control to the policy"
        ),
    )
    parser.add_argument(
        "--hg-label-outcome", dest="hg_label_outcome", action="store_true",
        help=("label each ACT+human HG trial with 1=success or 0=failure; "
              "require a manual R/Y reset before the next trial; does not run RL"),
    )
    parser.add_argument(
        "--hil-manual-outcome", dest="hg_label_outcome", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--hg-leader-interface", default="socketcan")
    parser.add_argument("--hg-leader-channel", default="can_leader1")
    parser.add_argument(
        "--hg-leader-init-timeout", type=float, default=30.0,
        help="seconds to wait for leader feedback/drag input",
    )
    parser.add_argument(
        "--hg-dataset-root", type=Path, default=PIPER_RAW_ROOT,
        help="piper_raw HG-DAgger root; next episode encoders are prewarmed here",
    )
    parser.add_argument(
        "--hg-task", default=None,
        help="task label written to HG-DAgger episodes (required for ACT)",
    )
    parser.add_argument("--hg-setup-id", default="online_policy_hg_dagger_v1")
    parser.add_argument("--hg-setup-note", default="")
    parser.add_argument("--hil-run", type=Path, help="prepared ACT-residual SAC run directory")
    parser.add_argument("--hil-min-updates", type=int, default=100)
    parser.add_argument("--hil-deterministic", action="store_true", default=True)
    parser.add_argument("--hil-explore", dest="hil_deterministic", action="store_false",
                        help="explicitly enable stochastic residual exploration")
    parser.add_argument("--hil-max-action-age-ms", type=float, default=100.0,
                        help="HIL observation-to-command deadline including inference; camera acquisition age is checked separately")
    parser.add_argument("--hg-record-fps", type=int, default=None)
    parser.add_argument(
        "--hg-start-joint-motion-deg",
        type=float,
        default=0.75,
        help="leader-relative joint motion required to start expert control",
    )
    parser.add_argument(
        "--hg-start-gripper-motion-mm",
        type=float,
        default=1.5,
        help="leader-relative gripper motion required to start expert control",
    )
    parser.add_argument(
        "--hg-start-motion-frames",
        type=int,
        default=2,
        help="consecutive moved samples required to start expert control",
    )
    parser.add_argument(
        "--hg-joint-signs", type=int, nargs=6,
        default=[1, 1, 1, 1, 1, 1],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument(
        "--hg-joint-offsets-deg", type=float, nargs=6,
        default=[0.0] * 6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default=cfg["device"])
    parser.add_argument("--rate", type=float, default=cfg["rate"])
    parser.add_argument(
        "--duration", type=float, default=cfg["duration"],
        help=(
            "shadow-mode wall-clock duration; in execute mode, maximum "
            "duration of each X-started trial"
        ),
    )
    parser.add_argument("--max-inference-ms", type=float, default=cfg["max-inference-ms"])
    parser.add_argument(
        "--warn-inference-ms", type=float, default=cfg["warn-inference-ms"],
        help=(
            "soft synchronous-inference threshold in ms; exceeding it is "
            "logged and execution continues until --max-inference-ms"
        ),
    )
    parser.add_argument(
        "--max-chunk-inference-ms", type=float, default=None,
        help=(
            "limit for calls that generate a new action chunk (async-planned "
            "policies: SmolVLA, Diffusion, Pi0); defaults to n-action-steps/rate "
            "seconds"
        ),
    )
    parser.add_argument(
        "--max-smolvla-plan-age-ms", "--max-plan-age-ms", type=float,
        default=cfg["max-smolvla-plan-age-ms"],
        help=(
            "maximum age of the observation used to generate an async "
            "action before that action is rejected (async-planned policies)"
        ),
    )
    parser.add_argument(
        "--smolvla-replan-lead-steps", "--replan-lead-steps", type=int, default=None,
        help=(
            "request the next async chunk when this many actions remain; "
            "defaults to enough actions to cover max chunk inference "
            "(async-planned policies)"
        ),
    )
    parser.add_argument("--max-camera-age-ms", type=float, default=cfg["max-camera-age-ms"])
    parser.add_argument("--warmup-steps", type=int, default=cfg["warmup-steps"])
    parser.add_argument(
        "--task", default=None,
        help="required language instruction for SmolVLA, Pi0 or Pi0.5",
    )
    parser.add_argument(
        "--n-action-steps", "--n_action_steps", dest="n_action_steps",
        type=int, default=None,
        help=(
            "queued actions to execute before predicting a new chunk; "
            "defaults to the checkpoint value"
        ),
    )
    parser.add_argument(
        "--num-inference-steps", "--num_inference_steps",
        dest="num_inference_steps", type=int, default=None,
        help=(
            "Diffusion denoising / Pi0 flow steps per predicted chunk; defaults to the "
            "checkpoint value"
        ),
    )
    parser.add_argument(
        "--smolvla-num-steps", type=int, default=None,
        help=(
            "SmolVLA flow-matching integration steps; defaults to the "
            "checkpoint value"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff", "--temporal_ensemble_coeff",
        dest="temporal_ensemble_coeff", type=float, default=None,
        help=(
            "enable ACT temporal ensembling with this positive coefficient; "
            "requires n-action-steps=1"
        ),
    )
    parser.add_argument("--max-joint-step-deg", type=float, default=cfg["max-joint-step-deg"])
    parser.add_argument(
        "--max-joint-acceleration-deg-s2", type=float,
        default=cfg["max-joint-acceleration-deg-s2"],
        help=(
            "joint command acceleration limit; 0 disables it "
            "(ACT default: 200 deg/s^2)"
        ),
    )
    parser.add_argument("--max-gripper-step-mm", type=float, default=cfg["max-gripper-step-mm"])
    parser.add_argument("--max-joint-lead-deg", type=float, default=cfg["max-joint-lead-deg"])
    parser.add_argument("--max-gripper-lead-mm", type=float, default=cfg["max-gripper-lead-mm"])
    parser.add_argument(
        "--warn-act-raw-joint-gap-deg", type=float,
        default=cfg["warn-act-raw-joint-gap-deg"],
        help=(
            "ACT raw joint-gap warning threshold; the safely limited target "
            "continues to execute"
        ),
    )
    parser.add_argument(
        "--max-act-raw-joint-gap-deg", type=float,
        default=cfg["max-act-raw-joint-gap-deg"],
        help=(
            "ACT raw target-to-feedback severe warning threshold; ACT uses "
            "a frozen policy-clock backpressure target instead of stopping "
            "on this telemetry alone"
        ),
    )
    parser.add_argument(
        "--max-act-raw-gap-consecutive-steps", type=int,
        default=cfg["max-act-raw-gap-consecutive-steps"],
        help=(
            "consecutive ticks above the severe ACT raw joint-gap threshold "
            "reported as telemetry (does not itself stop fixed-subgoal tracking)"
        ),
    )
    parser.add_argument(
        "--max-act-raw-gripper-gap-mm", type=float,
        default=cfg["max-act-raw-gripper-gap-mm"],
        help=(
            "ACT raw gripper-gap warning threshold; Pi0 retains it as a "
            "raw-action watchdog"
        ),
    )
    parser.add_argument(
        "--max-act-raw-joint-bound-tolerance-deg", type=float,
        default=cfg["max-act-raw-joint-bound-tolerance-deg"],
        help=(
            "small ACT raw joint-target overshoot allowed beyond the software "
            "bounds; the executable target is still hard-clamped to the bounds"
        ),
    )
    parser.add_argument(
        "--max-act-raw-gripper-bound-tolerance-mm", type=float,
        default=cfg["max-act-raw-gripper-bound-tolerance-mm"],
        help="ACT raw gripper target tolerance beyond [0, gripper-max-mm]",
    )
    parser.add_argument(
        "--max-act-subgoal-no-progress-steps", "--max-act-clamp-hold-steps",
        dest="max_act_clamp_hold_steps", type=int,
        default=cfg["max-act-clamp-hold-steps"],
        help=(
            "maximum consecutive ticks without feedback progress toward a "
            "hard-clamped ACT backpressure target (old "
            "--max-act-clamp-hold-steps is an alias)"
        ),
    )
    # Compatibility only: the first fixed-point implementation exposed these
    # flags.  Backpressure now releases on the final limiter state, so exact
    # position tolerances no longer participate in control.
    parser.add_argument(
        "--act-subgoal-joint-tolerance-deg", type=float, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--act-subgoal-gripper-tolerance-mm", type=float, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--act-subgoal-min-progress-deg", type=float,
        default=cfg["act-subgoal-min-progress-deg"],
        help="minimum joint-error reduction that resets the no-progress counter",
    )
    parser.add_argument(
        "--act-subgoal-min-gripper-progress-mm", type=float,
        default=cfg["act-subgoal-min-gripper-progress-mm"],
        help="minimum gripper-error reduction that resets the no-progress counter",
    )
    parser.add_argument("--gripper-max-mm", type=float, default=cfg["gripper-max-mm"])
    parser.add_argument("--gripper-force", type=float, default=cfg["gripper-force"])
    parser.add_argument("--report", type=Path, default=None,
                        help="遥测报告输出；默认按 checkpoint 自动生成路径")
    parser.add_argument("--approval", type=Path, default=None,
                        help="审批文件；默认按 checkpoint 自动查找")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--experimental-smolvla-execute", action="store_true",
        help="enable the separately approved conservative SmolVLA execution path",
    )
    parser.add_argument(
        "--keyboard-hold-seconds", type=float, default=cfg["keyboard-hold-seconds"],
        help=(
            "seconds of repeated Y key events required to home"
        ),
    )
    parser.add_argument(
        "--home-speed-percent", type=base.percentage, default=cfg["home-speed-percent"],
        help="MOVE_J speed percentage for ACT startup and R/Y homing",
    )
    parser.add_argument(
        "--home-timeout", type=float, default=cfg["home-timeout"],
        help="ACT startup and keyboard-Y homing timeout in seconds",
    )
    base.add_connection_arguments(parser, cfg)
    return parser.parse_args(argv)


def validate_motion_limits(args, policy_type=None):
    """Validate base limits and any policy-specific limits after overrides."""
    if args.max_joint_step_deg <= 0 or args.max_gripper_step_mm <= 0:
        raise RuntimeError("step limits must be positive")
    if (
        args.max_joint_acceleration_deg_s2 < 0
        or not math.isfinite(args.max_joint_acceleration_deg_s2)
    ):
        raise RuntimeError(
            "max-joint-acceleration-deg-s2 must be finite and non-negative"
        )
    if args.max_joint_lead_deg < args.max_joint_step_deg:
        raise RuntimeError("max-joint-lead-deg must be at least max-joint-step-deg")
    if args.max_gripper_lead_mm < args.max_gripper_step_mm:
        raise RuntimeError("max-gripper-lead-mm must be at least max-gripper-step-mm")
    if args.max_act_clamp_hold_steps < 1:
        raise RuntimeError("max-act-subgoal-no-progress-steps must be positive")
    if args.max_act_raw_gap_consecutive_steps < 1:
        raise RuntimeError("max-act-raw-gap-consecutive-steps must be positive")
    for name in ("max_act_raw_joint_gap_deg", "max_act_raw_gripper_gap_mm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise RuntimeError(name.replace("_", "-") + " must be positive")
    if (
        args.max_act_raw_joint_bound_tolerance_deg < 0
        or not math.isfinite(args.max_act_raw_joint_bound_tolerance_deg)
    ):
        raise RuntimeError(
            "max-act-raw-joint-bound-tolerance-deg must be finite "
            "and non-negative"
        )
    if (
        args.max_act_raw_gripper_bound_tolerance_mm < 0
        or not math.isfinite(args.max_act_raw_gripper_bound_tolerance_mm)
    ):
        raise RuntimeError(
            "max-act-raw-gripper-bound-tolerance-mm must be finite "
            "and non-negative"
        )
    for name in (
        "act_subgoal_min_progress_deg",
        "act_subgoal_min_gripper_progress_mm",
    ):
        value = getattr(args, name)
        if value < 0 or not math.isfinite(value):
            raise RuntimeError(
                name.replace("_", "-") + " must be finite and non-negative"
            )
    if policy_type in {"act", "pi0"} and (
        args.max_act_raw_joint_gap_deg is None
        or args.max_act_raw_gripper_gap_mm is None
    ):
        raise RuntimeError("ACT raw-action gap limits must be configured")
    if (
        policy_type == "act"
        and args.warn_act_raw_joint_gap_deg is not None
        and not 0 < args.warn_act_raw_joint_gap_deg
        <= args.max_act_raw_joint_gap_deg
    ):
        raise RuntimeError(
            "warn-act-raw-joint-gap-deg must be positive and no greater "
            "than max-act-raw-joint-gap-deg"
        )


def validate_approval(
    path: Path, digest: str, policy_type: str, runtime: dict | None = None
) -> None:
    if not path.is_file():
        raise RuntimeError(f"execution locked: missing shadow approval {path}")
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("checkpoint_sha256") != digest or approval.get("approved") is not True:
        raise RuntimeError("execution locked: approval does not match this checkpoint")
    if policy_type not in ("act", "smolvla", "diffusion", "pi0"):
        return
    if approval.get("policy_type") != policy_type:
        raise RuntimeError(
            "execution locked: approval is not policy-bound"
        )
    if (
        policy_type in {"act", "pi0"}
        and approval.get("act_raw_action_watchdog_passed") is not True
    ):
        raise RuntimeError(
            "execution locked: ACT approval predates raw-action shadow "
            "validation; run validate-runtime again"
        )
    approved_runtime = approval.get("runtime")
    if not isinstance(approved_runtime, dict) or runtime is None:
        raise RuntimeError("execution locked: approval has no runtime binding")
    for key, expected in runtime.items():
        actual = approved_runtime.get(key)
        if isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and math.isclose(
                    float(actual), expected, rel_tol=1e-6, abs_tol=1e-6
                )
            )
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                "execution locked: approval runtime mismatch for "
                f"{key}: approved={actual!r}, requested={expected!r}"
            )


def load_dataset_metadata(checkpoint: Path):
    """Load the dataset schema recorded with the checkpoint, without frames.

    Falls back to the workspace dataset root for legacy checkpoints trained
    before the datasets were renamed (``piper_pick_up*`` → ``piper_pick_cup*``)
    and moved out of the pre-refactor ``/home/dm/data`` tree. Inaccessible
    training-machine paths (e.g. a cloud server's /root) also permit fallback.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from ..paths import LEROBOT_DATA_ROOT

    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.is_file():
        raise RuntimeError(f"checkpoint is missing {train_config_path.name}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    dataset_config = train_config.get("dataset") or {}
    repo_id = dataset_config.get("repo_id")
    root_value = dataset_config.get("root")
    if not repo_id or not root_value:
        raise RuntimeError("checkpoint train_config has no dataset repo_id/root")

    recorded_root = Path(root_value).expanduser()
    try:
        recorded_root = recorded_root.resolve()
    except PermissionError:
        # Resolving symlinks can itself fail before we probe meta/info.json.
        recorded_root = recorded_root.absolute()
    name = recorded_root.name
    candidates = [
        recorded_root,
        LEROBOT_DATA_ROOT / name,
        LEROBOT_DATA_ROOT / name.replace("piper_pick_up", "piper_pick_cup"),
    ]
    inaccessible = []
    for dataset_root in dict.fromkeys(candidates):
        try:
            has_metadata = (dataset_root / "meta" / "info.json").is_file()
        except PermissionError:
            inaccessible.append(str(dataset_root))
            logging.getLogger(__name__).warning(
                "Dataset path is inaccessible; trying local fallback: %s", dataset_root
            )
            continue
        if has_metadata:
            if dataset_root != recorded_root and "piper_pick_up" in repo_id:
                repo_id = repo_id.replace("piper_pick_up", "piper_pick_cup")
            if dataset_root != recorded_root:
                print(f"[dataset] using local metadata: {dataset_root}")
            return LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    raise RuntimeError(
        "checkpoint dataset metadata not found: "
        f"{recorded_root} (also tried {candidates[1]} and {candidates[2]})"
        + (f"; permission denied: {', '.join(inaccessible)}" if inaccessible else "")
    )


def main(argv: list[str] | None = None) -> int:
    # Covers HIL, HG and plain ACT, including failures/interrupts in recorder,
    # camera and arm cleanup after the key reader has already been closed.
    with keyboard_api.preserve_terminal_settings():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    explicitly_set = {
        token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        for token in raw_argv
        if token.startswith("--")
    }
    for alias, destination in {
        "max_plan_age_ms": "max_smolvla_plan_age_ms",
        "replan_lead_steps": "smolvla_replan_lead_steps",
        "max_act_subgoal_no_progress_steps": "max_act_clamp_hold_steps",
    }.items():
        if alias in explicitly_set:
            explicitly_set.add(destination)
    if not 1 <= args.rate <= 20 or args.duration <= 0:
        raise RuntimeError("rate must be 1..20 Hz and duration must be positive")
    if not 1 <= args.camera_fps <= 60:
        raise RuntimeError("camera-fps must be 1..60")
    if args.recording_fps is not None and not 1 <= args.recording_fps <= 60:
        raise RuntimeError("recording-fps must be 1..60")
    if args.recording_encoder_queue_size < 1:
        raise RuntimeError("recording-encoder-queue-size must be positive")
    recording_fps = args.recording_fps or max(
        1, int(round(min(args.rate, args.camera_fps)))
    )
    hg_record_fps = args.hg_record_fps or max(
        1, int(round(min(args.rate, args.camera_fps)))
    )
    if not 1 <= hg_record_fps <= 60:
        raise RuntimeError("hg-record-fps must be 1..60")
    if (
        not math.isfinite(args.hg_start_joint_motion_deg)
        or args.hg_start_joint_motion_deg <= 0
    ):
        raise RuntimeError("hg-start-joint-motion-deg must be positive")
    if (
        not math.isfinite(args.hg_start_gripper_motion_mm)
        or args.hg_start_gripper_motion_mm <= 0
    ):
        raise RuntimeError("hg-start-gripper-motion-mm must be positive")
    if not 1 <= args.hg_start_motion_frames <= 20:
        raise RuntimeError("hg-start-motion-frames must be 1..20")
    if args.hg_dagger and not args.execute:
        raise RuntimeError("--hg-dagger requires --execute")
    if args.hg_label_outcome and not args.hg_dagger:
        raise RuntimeError("--hg-label-outcome requires --hg-dagger")
    if args.hil_run and args.execute and not (args.hg_dagger and args.hg_label_outcome):
        raise RuntimeError("HIL execution requires expert takeover and explicit outcome labels")
    if not math.isfinite(args.hil_max_action_age_ms) or not 0 < args.hil_max_action_age_ms <= 500:
        raise RuntimeError("hil-max-action-age-ms must be within (0, 500]")
    if args.hg_dagger and str(args.hg_leader_channel) == str(args.channel):
        raise RuntimeError("HG-DAgger leader and follower CAN channels must differ")
    if args.hg_leader_init_timeout <= 0:
        raise RuntimeError("hg-leader-init-timeout must be positive")
    recording_root = args.recording_root.expanduser()
    if not recording_root.is_absolute():
        recording_root = WORKSPACE_ROOT / recording_root
    recording_root = recording_root.resolve()
    hg_dataset_root = args.hg_dataset_root.expanduser()
    if not hg_dataset_root.is_absolute():
        hg_dataset_root = WORKSPACE_ROOT / hg_dataset_root
    hg_dataset_root = hg_dataset_root.resolve()
    if not 1 <= args.warmup_steps <= 20:
        raise RuntimeError("warmup-steps must be 1..20")
    if not 0.2 <= args.keyboard_hold_seconds <= 5.0:
        raise RuntimeError("keyboard-hold-seconds must be in [0.2, 5.0]")
    if args.home_timeout <= 0:
        raise RuntimeError("home-timeout must be positive")
    if args.max_inference_ms <= 0 or args.max_camera_age_ms <= 0:
        raise RuntimeError("inference and camera-age limits must be positive")
    if args.warn_inference_ms <= 0:
        raise RuntimeError("warn-inference-ms must be positive")
    if (
        args.max_chunk_inference_ms is not None
        and args.max_chunk_inference_ms <= 0
    ):
        raise RuntimeError("max-chunk-inference-ms must be positive")
    if args.max_smolvla_plan_age_ms <= 0:
        raise RuntimeError("max-smolvla-plan-age-ms must be positive")
    validate_motion_limits(args)
    checkpoint = args.checkpoint.expanduser().resolve()

    system_fonts = Path("/usr/share/fonts/truetype/dejavu")
    import cv2
    # opencv-python overwrites QT_QPA_FONTDIR during import with its removed
    # cv2/qt/fonts directory.  Restore a real system font directory afterwards,
    # before RoleCameras creates the first Qt window.
    if system_fonts.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(system_fonts)
    import numpy as np
    import torch
    from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    # Diffusers 0.39 probes a removed legacy TorchAO tensor path and emits a
    # warning even for ordinary FP32 ACT checkpoints.  The compatible cu128
    # TorchAO wheel is installed and its APIs are usable; silence only that
    # irrelevant optional-checkpoint probe while importing LeRobot's factory.
    torchao_probe_logger = logging.getLogger(
        "diffusers.quantizers.torchao.torchao_quantizer"
    )
    previous_torchao_probe_level = torchao_probe_logger.level
    torchao_probe_logger.setLevel(logging.ERROR)
    try:
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    finally:
        torchao_probe_logger.setLevel(previous_torchao_probe_level)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the requested device but is unavailable")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_type = getattr(config, "type", None)
    checkpoint_n_action_steps = getattr(config, "n_action_steps", None)
    checkpoint_temporal_ensemble_coeff = getattr(
        config, "temporal_ensemble_coeff", None
    )
    if policy_type not in {"act", "diffusion", "smolvla", "pi05", "pi0"}:
        raise RuntimeError(
            "online_policy supports ACT, Diffusion, gated SmolVLA, and "
            "shadow-only Pi0.5 "
            f"checkpoints; got {policy_type!r}"
        )
    if policy_type == "pi05":
        # A base Pi0.5 action has 32 unnamed dimensions and no Piper
        # postprocessor statistics. A shadow approval cannot unlock CAN.
        if args.execute:
            raise RuntimeError(
                "Pi0.5 CAN execution is locked: no Piper-finetuned 7D "
                "checkpoint, action contract, or inverse-normalization "
                "statistics are available. Remove --execute for shadow mode."
            )
        if args.experimental_smolvla_execute:
            raise RuntimeError("--experimental-smolvla-execute is not for Pi0.5")
        task = args.task.strip() if args.task is not None else ""
        if not task:
            raise RuntimeError("Pi0.5 shadow requires a non-empty --task")
        digest = checkpoint_digest(checkpoint)
        if args.report is None:
            args.report = runtime_artifact_paths(checkpoint, digest)[0]
        from . import pi05_shadow
        return pi05_shadow.run(args, config, checkpoint, digest, task)
    if args.hg_label_outcome and policy_type != "act":
        raise RuntimeError("--hg-label-outcome currently requires an ACT checkpoint")
    if policy_type == "pi0":
        torch.set_num_threads(4)
        if args.device != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Pi0 LoRA runtime requires CUDA with BF16 support")
        if ("action" not in config.output_features
            or tuple(config.output_features["action"].shape) != (7,)
            or tuple(config.input_features["observation.state"].shape) != (7,)
            or config.use_relative_actions):
            raise RuntimeError("Pi0 runtime requires Piper 7D absolute joint actions")
        if args.execute and not args.experimental_pi0_execute:
            raise RuntimeError(
                "Pi0 execution requires --experimental-pi0-execute"
            )
    elif args.experimental_pi0_execute or args.base_checkpoint is not None:
        raise RuntimeError("Pi0-specific options require a Pi0 checkpoint")
    expected = set(config.input_features)
    if policy_type == "pi0":
        expected -= {f"observation.images.empty_camera_{i}" for i in range(config.empty_cameras)}
    camera_roles = required_camera_roles(expected)
    camera_serials = {"global": args.camera_serial}
    if "wrist" in camera_roles:
        if not args.wrist_camera_serial:
            raise RuntimeError(
                "checkpoint requires observation.images.wrist; provide --wrist-camera-serial"
            )
        if args.wrist_camera_serial == args.camera_serial:
            raise RuntimeError("global and wrist camera serials must be different")
        camera_serials["wrist"] = args.wrist_camera_serial
    if not sys.stdin.isatty():
        raise RuntimeError("online keyboard control requires an interactive TTY")
    pi0_base = (args.base_checkpoint or ARTIFACTS_ROOT / "pretrained/pi0_base").resolve()
    digest = pi0_checkpoint_digest(checkpoint, pi0_base) if policy_type == "pi0" else checkpoint_digest(checkpoint)
    # SmolVLA and Diffusion run their inference in a background planner so
    # the control tick never blocks on CUDA. ACT's forward pass is fast
    # enough to stay synchronous.
    async_planned = policy_type in {"smolvla", "diffusion", "pi0"}
    for key, value in policy_overrides(policy_type).items():
        # YAML keys are hyphenated (CLI-flag style); argparse dests are
        # underscored. Convert here so the override lands on the real
        # attribute instead of creating a junk dashed one.
        destination = key.replace("-", "_")
        if destination not in explicitly_set:
            setattr(args, destination, value)
    # Per-policy defaults are applied after argparse, so validate again here.
    # This catches contradictory overrides before an approval or robot command.
    if not 0 < args.warn_inference_ms <= args.max_inference_ms:
        raise RuntimeError(
            "warn-inference-ms must be positive and no greater than "
            "max-inference-ms"
        )
    validate_motion_limits(args, policy_type)
    shadow_report, execute_report, generated_approval = runtime_artifact_paths(
        checkpoint, digest
    )
    if args.report is None:
        args.report = execute_report if args.execute else shadow_report
    if args.approval is None:
        args.approval = find_approval(checkpoint, digest, policy_type)
    task = args.task.strip() if args.task is not None else None
    if policy_type in {"smolvla", "pi0"}:
        if not task:
            raise RuntimeError(f"{policy_type} requires a non-empty --task")
    elif task is not None:
        raise RuntimeError("--task is only supported by language-conditioned checkpoints")
    hg_task = args.hg_task.strip() if args.hg_task is not None else task
    if args.hg_dagger and not hg_task:
        raise RuntimeError(
            "--hg-dagger requires --hg-task for the correction dataset label"
        )
    if args.experimental_smolvla_execute and policy_type != "smolvla":
        raise RuntimeError(
            "--experimental-smolvla-execute requires a SmolVLA checkpoint"
        )
    if args.n_action_steps is not None:
        if not hasattr(config, "n_action_steps"):
            raise RuntimeError("this policy does not support --n-action-steps")
        config.n_action_steps = args.n_action_steps
    if args.num_inference_steps is not None:
        if policy_type not in {"diffusion", "pi0"} or not hasattr(config, "num_inference_steps"):
            raise RuntimeError(
                "--num-inference-steps requires Diffusion or Pi0"
            )
        maximum = config.num_train_timesteps if policy_type == "diffusion" else 100
        if not 1 <= args.num_inference_steps <= maximum:
            raise RuntimeError(
                f"num-inference-steps must be between 1 and {maximum}"
            )
        config.num_inference_steps = args.num_inference_steps
    if args.smolvla_num_steps is not None:
        if policy_type != "smolvla" or not hasattr(config, "num_steps"):
            raise RuntimeError(
                "--smolvla-num-steps is only supported by SmolVLA checkpoints"
            )
        if not 1 <= args.smolvla_num_steps <= 100:
            raise RuntimeError("smolvla-num-steps must be in [1, 100]")
        config.num_steps = args.smolvla_num_steps
    if args.temporal_ensemble_coeff is not None:
        if not hasattr(config, "temporal_ensemble_coeff"):
            raise RuntimeError(
                "this policy does not support --temporal-ensemble-coeff"
            )
        if args.temporal_ensemble_coeff <= 0:
            raise RuntimeError("temporal-ensemble-coeff must be positive")
        config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        if args.n_action_steps is None:
            config.n_action_steps = 1
    if hasattr(config, "n_action_steps"):
        if policy_type == "diffusion":
            max_action_steps = config.horizon - config.n_obs_steps + 1
        else:
            max_action_steps = config.chunk_size
        if not 1 <= config.n_action_steps <= max_action_steps:
            raise RuntimeError(
                f"n-action-steps must be in [1, {max_action_steps}], got "
                f"{config.n_action_steps}"
            )
        if (
            getattr(config, "temporal_ensemble_coeff", None) is not None
            and config.n_action_steps != 1
        ):
            raise RuntimeError(
                "temporal ensembling requires --n-action-steps=1"
            )
    if async_planned:
        max_chunk_inference_ms = (
            args.max_chunk_inference_ms
            if args.max_chunk_inference_ms is not None
            else 1000.0 * config.n_action_steps / args.rate
        )
        chunk_coverage_ms = 1000.0 * config.n_action_steps / args.rate
        if max_chunk_inference_ms > chunk_coverage_ms:
            raise RuntimeError(
                "max-chunk-inference-ms must not exceed the action chunk "
                "coverage time"
            )
        replan_lead_steps = (
            args.smolvla_replan_lead_steps
            if args.smolvla_replan_lead_steps is not None
            else min(
                config.n_action_steps,
                max(1, math.ceil(max_chunk_inference_ms * args.rate / 1000.0)),
            )
        )
        if not 1 <= replan_lead_steps <= config.n_action_steps:
            raise RuntimeError(
                "smolvla-replan-lead-steps must be in "
                f"[1, {config.n_action_steps}]"
            )
    else:
        if args.max_chunk_inference_ms is not None:
            raise RuntimeError(
                "--max-chunk-inference-ms is only supported by async-planned "
                "checkpoints (SmolVLA, Diffusion, Pi0)"
            )
        max_chunk_inference_ms = args.max_inference_ms
        replan_lead_steps = None
    if args.execute:
        if policy_type == "smolvla":
            if not args.experimental_smolvla_execute:
                raise RuntimeError(
                    "SmolVLA execution requires --experimental-smolvla-execute"
                )
            if config.n_action_steps < 2:
                raise RuntimeError(
                    "async SmolVLA execution requires --n-action-steps >=2"
                )
            if (
                args.max_joint_step_deg > 0.25
                or args.max_gripper_step_mm > 0.5
                or args.max_joint_lead_deg > 0.5
                or args.max_gripper_lead_mm > 1.0
            ):
                raise RuntimeError(
                    "SmolVLA execution requires conservative limits: joint "
                    "step/lead <=0.25/0.5 deg and gripper step/lead <=0.5/1.0 mm"
                )
        approval_runtime = {
            "device": args.device,
            "interface": args.interface,
            "channel": str(args.channel),
            "firmware": args.firmware,
            "camera_serial": args.camera_serial,
            **({"camera_serials": camera_serials} if "wrist" in camera_roles else {}),
            "camera_fps": args.camera_fps,
            "task": task,
            **({"pi0_clock_aligned": True,
                "num_inference_steps": config.num_inference_steps} if policy_type == "pi0" else {}),
            **({
                "chunk_size": getattr(config, "chunk_size", None),
                "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
            } if policy_type in {"act", "pi0"} else {}),
            "n_action_steps": getattr(config, "n_action_steps", None),
            "num_flow_steps": getattr(config, "num_steps", None),
            "rate_hz": float(args.rate),
            "planner_mode": "async_chunked" if async_planned else None,
            "max_chunk_inference_ms": float(max_chunk_inference_ms),
            "max_smolvla_plan_age_ms": (
                float(args.max_smolvla_plan_age_ms)
                if async_planned else None
            ),
            "smolvla_replan_lead_steps": replan_lead_steps,
            "max_joint_step_deg": float(args.max_joint_step_deg),
            "max_joint_acceleration_deg_s2": float(
                args.max_joint_acceleration_deg_s2
            ),
            "max_gripper_step_mm": float(args.max_gripper_step_mm),
            "max_joint_lead_deg": float(args.max_joint_lead_deg),
            "max_gripper_lead_mm": float(args.max_gripper_lead_mm),
            **({
                "warn_act_raw_joint_gap_deg": (
                    None if args.warn_act_raw_joint_gap_deg is None
                    else float(args.warn_act_raw_joint_gap_deg)
                ),
                "max_act_raw_joint_gap_deg": float(
                    args.max_act_raw_joint_gap_deg
                ),
                "max_act_raw_gap_consecutive_steps": (
                    args.max_act_raw_gap_consecutive_steps
                ),
                "max_act_raw_gripper_gap_mm": float(
                    args.max_act_raw_gripper_gap_mm
                ),
                "max_act_raw_joint_bound_tolerance_deg": float(
                    args.max_act_raw_joint_bound_tolerance_deg
                ),
                "max_act_raw_gripper_bound_tolerance_mm": float(
                    args.max_act_raw_gripper_bound_tolerance_mm
                ),
                "max_act_clamp_hold_steps": args.max_act_clamp_hold_steps,
            } if policy_type in {"act", "pi0"} else {}),
            **({
                "act_subgoal_min_progress_deg": float(
                    args.act_subgoal_min_progress_deg
                ),
                "act_subgoal_min_gripper_progress_mm": float(
                    args.act_subgoal_min_gripper_progress_mm
                ),
            } if policy_type == "act" else {}),
            "gripper_max_mm": float(args.gripper_max_mm),
            "gripper_force": float(args.gripper_force),
            "max_camera_age_ms": float(args.max_camera_age_ms),
        }
        approval_checked = validate_execution_approval(
            args.approval.resolve(), digest, policy_type, approval_runtime
        )
        if not approval_checked:
            if policy_type == "pi0":
                print(
                    "[approval] Pi0 execution does not require a prior "
                    "shadow approval",
                    flush=True,
                )
            else:
                print(
                    "[approval] disabled by modeling.yaml; skipping approval check",
                    flush=True,
                )
    config.pretrained_path = str(checkpoint)
    config.device = args.device
    dataset_metadata = load_dataset_metadata(checkpoint)
    metadata_inputs = set(dataset_metadata.features) & expected
    if metadata_inputs != expected:
        raise RuntimeError(
            "checkpoint and dataset metadata disagree on policy inputs: "
            f"checkpoint={sorted(expected)}, metadata={sorted(metadata_inputs)}"
        )
    training_fps = int(dataset_metadata.fps)
    if policy_type == "pi0":
        from .pi0_lora import validate_dataset
        validate_dataset(Path(dataset_metadata.root))
        if not math.isclose(args.rate, training_fps, abs_tol=1e-6):
            raise RuntimeError("Pi0 clock alignment requires rate to match dataset FPS")
    if not math.isclose(args.rate, training_fps, abs_tol=1e-6):
        print(
            f"[policy] WARNING: control rate {args.rate:g} Hz differs from "
            f"training dataset rate {training_fps} Hz.", flush=True,
        )
    if args.camera_fps != training_fps:
        print(
            f"[policy] WARNING: camera sampling target {args.camera_fps} FPS differs "
            f"from training dataset rate {training_fps} FPS.", flush=True,
        )
    if policy_type == "pi0":
        from .pi0_lora_check import load_pi0_lora
        policy = load_pi0_lora(checkpoint, pi0_base, config, args.device)
    else:
        policy = make_policy(config, ds_meta=dataset_metadata)
    policy.eval()
    if hasattr(config, "n_action_steps"):
        print(
            f"Policy execution: type={policy_type}, "
            f"chunk_size={getattr(config, 'chunk_size', None)}, "
            f"n_action_steps={config.n_action_steps} "
            f"(checkpoint={checkpoint_n_action_steps}), "
            f"rate={args.rate:g}Hz/{training_fps}Hz_train, "
            "joint_acceleration_limit="
            f"{args.max_joint_acceleration_deg_s2:g}deg/s^2, "
            f"cameras={'+'.join(camera_roles)}@{args.camera_fps}FPS, "
            "temporal_ensemble_coeff="
            f"{getattr(config, 'temporal_ensemble_coeff', None)} "
            f"(checkpoint={checkpoint_temporal_ensemble_coeff}), "
            "num_inference_steps="
            f"{getattr(config, 'num_inference_steps', None)}, "
            f"num_flow_steps={getattr(config, 'num_steps', None)}, "
            f"max_chunk_inference_ms={max_chunk_inference_ms:.1f}",
            flush=True,
        )
    if policy_type == "act" and (
        config.n_action_steps != checkpoint_n_action_steps
        or config.temporal_ensemble_coeff != checkpoint_temporal_ensemble_coeff
    ):
        print(
            "[policy] WARNING: ACT inference cadence/temporal ensembling differs "
            "from the checkpoint training configuration.", flush=True,
        )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    hil_runtime = None
    hil_can_guard = None
    if args.hil_run:
        if policy_type != "act":
            raise RuntimeError("HIL residual SAC requires an ACT prior")
        from .hil_serl.config import Config as HILConfig
        if HILConfig.load(args.hil_run / "config.json").format_version.endswith("v1"):
            raise RuntimeError("legacy HIL residual execution disabled; use hil-serl actor --mode baseline or prepare a v2 run")
        from .hil_serl.runtime import HILRuntime
        hil_runtime = HILRuntime(
            args.hil_run, checkpoint, policy, preprocessor, postprocessor,
            device=args.device, deterministic=args.hil_deterministic,
            minimum_updates=args.hil_min_updates,
        )
        if args.execute and args.hg_task != hil_runtime.cfg.task:
            raise RuntimeError("HIL task differs from the prepared replay task")
        if abs(args.gripper_max_mm / 1000 - hil_runtime.cfg.gripper_max_m) > 1e-8:
            raise RuntimeError("HIL gripper range differs from prepared replay")
        expected_serials = {role: str(camera_serials[role]) for role in hil_runtime.cfg.camera_roles}
        if hil_runtime.cfg.camera_serials and expected_serials != hil_runtime.cfg.camera_serials:
            raise RuntimeError("HIL camera serials differ from prepared replay")
        if args.execute and set(hil_runtime.cfg.camera_serials) != set(hil_runtime.cfg.camera_roles):
            raise RuntimeError("HIL camera identity missing; re-export replay with camera metadata before execution")
        if abs(args.rate - hil_runtime.cfg.rate_hz) > 1e-6:
            raise RuntimeError("HIL control rate differs from the prepared replay FPS")
        if args.execute and abs(hg_record_fps - args.rate) > 1e-6:
            raise RuntimeError("HIL RAW recording FPS must equal the control rate")

    robot = None
    cameras = None
    recorder = None
    hg_expert = None
    hg_recorder = None
    hg_intervening = False
    hg_recording = False
    hg_last_sample = None
    hg_started_at = None
    hg_takeover_action = None
    hg_expert_raw = None
    hg_intervention_id = -1
    hg_next_intervention_id = 0
    pending_outcome_episode = None
    trial_reset_required = False
    hg_motion_gate = TakeoverMotionGate(
        math.radians(args.hg_start_joint_motion_deg),
        args.hg_start_gripper_motion_mm / 1000.0,
        args.hg_start_motion_frames,
    )
    last_novice_action = None
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    keyboard = keyboard_api.TerminalKeyEventReader()
    key_tracker = KeyboardIntentTracker(args.keyboard_hold_seconds)
    command_gate = False
    motion_enabled = False
    arm_prepared = False
    command_target = None
    previous_joint_velocity = None
    held_act_action = None
    latched_act_subgoal = None
    act_subgoal_previous_joint_error = None
    act_subgoal_previous_gripper_error = None
    act_hold_steps = 0
    act_raw_gap_hard_steps = 0
    planner = None
    start_pending = False
    pending_plan_request_id = 0
    homing_active = False
    emergency_stopped = False
    record_blocking_motion = None
    # Piper JS is encoded as move_mode=J plus mit_mode=MIT. The controller's
    # mode_feedback therefore remains MOVE_J (1); MOVE_MIT (4/6) is reserved
    # for the separate per-joint MIT API.
    expected_js_mode = 1
    try:
        keyboard.start()
        if hil_runtime is not None and args.interface == "socketcan":
            from .hil_serl.safety import CANIdentityGuard
            guarded_channels = [str(args.channel)]
            if args.execute and args.hg_leader_interface == "socketcan":
                guarded_channels.append(str(args.hg_leader_channel))
            hil_can_guard = CANIdentityGuard(guarded_channels)
        robot, robot_config = base.connect_robot(args)
        base.ensure_arm_status_normal(robot, args.feedback_timeout)
        joint_bounds = deployment_joint_bounds(robot_config)
        if args.execute:
            # The SDK's canonical J2/J3 clamp does not account for this arm's
            # measured zero offset. Keep limits enforced here so the SDK cannot
            # turn a small negative target into an abrupt 0° target.
            robot.set_joint_limits_enabled(False)
            # Repeated automatic mode messages can prevent streamed JS targets
            # from taking effect. Mode changes are performed once, explicitly.
            robot.set_auto_set_motion_mode_enabled(False)
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        cameras = RoleCameras(
            camera_serials, 640, 480, args.camera_fps,
            preview=True,
            window_title="RobotArm Online Policy Cameras",
        )
        cameras.wait_ready()
        recorder = PolicyRecordingManager(
            recording_root,
            recording_fps,
            640,
            480,
            list(cameras.workers),
            cameras.metadata,
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "policy_type": policy_type,
                "task": task,
                "execute": bool(args.execute),
                "control_rate_hz": float(args.rate),
                "report": str(report),
            },
            args.recording_encoder_queue_size,
            cv2_module=cv2,
            np_module=np,
        )
        def homing_emergency_stop_requested():
            cameras.show(cameras.latest(), recorder.active)
            terminal_events = keyboard.poll()
            preview_keys = cameras.poll_keys()
            terminal_stop = any(key == "alt+s" for key, _ in terminal_events)
            preview_stop = any(
                key in (ord("s"), ord("S")) for key in preview_keys
            )
            if terminal_stop:
                print(
                    "\n[safety] Emergency stop requested by terminal Alt+S.",
                    flush=True,
                )
            if preview_stop:
                print(
                    "\n[safety] Emergency stop requested by camera-window S.",
                    flush=True,
                )
            return terminal_stop or preview_stop

        if args.hg_dagger:
            initial_leader_gripper_width = min(
                max(
                    base.wait_gripper_width(gripper, args.feedback_timeout),
                    0.0,
                ),
                args.gripper_max_mm / 1000.0,
            )
            print(
                f"[HG-DAgger] connecting expert leader {args.hg_leader_channel} "
                "before paired homing...",
                flush=True,
            )
            hg_expert = LeaderExpert(
                interface=args.hg_leader_interface,
                channel=str(args.hg_leader_channel),
                firmware=args.firmware,
                feedback_timeout=args.feedback_timeout,
                init_timeout=args.hg_leader_init_timeout,
                signs=args.hg_joint_signs,
                offsets_deg=args.hg_joint_offsets_deg,
                gripper_max=args.gripper_max_mm / 1000.0,
                gripper_force=args.gripper_force,
                initial_gripper_width=initial_leader_gripper_width,
            )

        if args.execute and policy_type == "act":
            # The ACT deployment starts from the demonstrated zero pose.  Do
            # this before warming the policy so its first live observation is
            # taken at home, and keep model control disarmed until X is pressed.
            home_gripper_width = (
                initial_leader_gripper_width
                if args.hg_dagger
                else min(
                    max(
                        base.wait_gripper_width(
                            gripper, args.feedback_timeout
                        ),
                        0.0,
                    ),
                    args.gripper_max_mm / 1000.0,
                )
            )
            print("ACT startup: enabling the arm and moving to home.", flush=True)
            homing_active = True
            home_arm_to_zero(
                robot, robot_config, gripper, home_gripper_width,
                args.gripper_force, args.feedback_timeout,
                args.home_speed_percent, args.home_timeout,
                homing_emergency_stop_requested,
                companion=hg_expert,
            )
            homing_active = False
        # CUDA and the policy allocate/compile lazily on the first call. Warm up
        # before watchdog accounting, then reset ACT's temporal/action state.
        warm_joints = [
            float(value) for value in base.wait_for_feedback(
                robot, args.feedback_timeout
            ).msg
        ]
        validate_joint_feedback(warm_joints, robot_config)
        warm_gripper = base.wait_gripper_width(gripper, args.feedback_timeout)
        warm_frames = cameras.latest()
        cameras.show(warm_frames, False)
        warm_images, _ = camera_batch_images(warm_frames, camera_roles, cv2, torch)
        warm_batch = {
            "observation.state": torch.from_numpy(
                np.asarray([*warm_joints, warm_gripper], dtype=np.float32)
            ).unsqueeze(0),
            **warm_images,
        }
        add_smolvla_task(warm_batch, policy_type, task)
        reset = getattr(policy, "reset", None)
        with torch.inference_mode():
            for _ in range(0 if policy_type == "pi0" else args.warmup_steps):
                if policy_type == "smolvla" and callable(reset):
                    # Force a real chunk generation on every warm-up call.
                    reset()
                if hil_runtime is not None:
                    hil_runtime.predict(warm_batch)
                else:
                    policy.select_action(preprocessor(warm_batch))
        if callable(reset):
            reset()
        if policy_type != "pi0":
            print(f"Policy warm-up: {args.warmup_steps} steps complete", flush=True)
        if args.execute and policy_type == "act":
            if homing_emergency_stop_requested():
                emergency_stopped = True
                robot.electronic_emergency_stop()
                print("\nAlt+S before X: electronic emergency stop sent.")
                return 130
            home_gripper_width = base.wait_gripper_width(
                gripper, args.feedback_timeout
            )
            prepare_js_control(
                robot, robot_config, gripper, home_gripper_width,
                args.gripper_force, args.feedback_timeout, expected_js_mode,
                ACT_STREAM_SPEED_PERCENT,
            )
            arm_prepared = True
            # A key pressed while homing or warming up must not arm motion.
            if homing_emergency_stop_requested():
                emergency_stopped = True
                robot.electronic_emergency_stop()
                print("\nAlt+S before X: electronic emergency stop sent.")
                return 130
            if args.hg_dagger:
                print(
                    "ACT follower and HG-DAgger leader are ready at home; "
                    "preparing the data recorder. Do not press X yet.",
                    flush=True,
                )
            else:
                print("ACT ready at home; press X to start model control.", flush=True)
        active_actions = deque()
        active_plan = None
        planner_requested = False
        if async_planned:
            planner = AsyncPlanner(
                policy, preprocessor, postprocessor, reset,
                config.n_action_steps,
                full_chunk=policy_type == "pi0", amp=policy_type == "pi0",
            )
            planner.start()
            if policy_type == "pi0":
                # Warm the actual worker/CUDA thread and autocast path. A cold
                # call can take >1s; its actions must never enter the live queue.
                for warmup_index in range(args.warmup_steps):
                    request_id = planner.request(warm_batch, time.monotonic())
                    warm_plan = planner.wait_ready(10.0, request_id)
                    if warm_plan is None or planner.failure is not None:
                        raise RuntimeError(f"Pi0 worker warm-up failed: {planner.failure or 'timeout'}")
                    cameras.show(cameras.latest(), False)
                    print(f"Pi0 worker warm-up {warmup_index + 1}/{args.warmup_steps}: "
                          f"{warm_plan['inference_ms']:.1f}ms (discarded)", flush=True)
                warm_frames = cameras.latest()
                warm_images, _ = camera_batch_images(warm_frames, camera_roles, cv2, torch)
                fresh_joints = read_valid_joints(robot, robot_config, args.feedback_timeout)
                fresh_gripper = base.wait_gripper_width(gripper, args.feedback_timeout)
                warm_batch = {"observation.state": torch.tensor([*fresh_joints, fresh_gripper],
                                                               dtype=torch.float32).unsqueeze(0),
                              **warm_images}
                add_smolvla_task(warm_batch, policy_type, task)
            requested_at = (min(time.monotonic(), *(f[1] for f in warm_frames.values()))
                            if policy_type == "pi0" else time.monotonic())
            pending_plan_request_id = planner.request(warm_batch, requested_at)
            active_plan = planner.wait_ready(
                max_chunk_inference_ms / 1000.0 + 2.0,
                pending_plan_request_id,
            )
            if active_plan is None:
                raise RuntimeError(
                    "async planner did not produce its initial chunk"
                )
            if planner.failure is not None:
                raise RuntimeError(f"async planner failed: {planner.failure}")
            if policy_type == "pi0":
                active_actions, _ = trim_pi0_plan(active_plan["actions"], active_plan,
                                                  time.monotonic(), args.rate, config.n_action_steps)
            else:
                active_actions.extend(active_plan["actions"])
        if args.hg_dagger:
            assert hg_expert is not None
            leader_follow_state = [
                *read_valid_joints(robot, robot_config, args.feedback_timeout),
                base.wait_gripper_width(gripper, args.feedback_timeout),
            ]
            if policy_type != "act":
                hg_expert.align_to_follower(
                    leader_follow_state,
                    speed_percent=args.home_speed_percent,
                    timeout=args.home_timeout,
                    stop_requested=homing_emergency_stop_requested,
                )
            else:
                hg_expert.resume_policy_control(leader_follow_state)
            hg_recorder = RawEpisodeRecorder(
                hg_dataset_root,
                hg_task,
                args.hg_setup_id,
                args.hg_setup_note,
                hg_record_fps,
                640,
                480,
                camera_serials,
                {
                    "model": "Piper",
                    "control_mode": "hil_residual_sac_move_js" if hil_runtime is not None else "human_gated_dagger_move_js",
                    "hil_run": str(hil_runtime.run) if hil_runtime is not None else None,
                    "hil_config_signature": hil_runtime.cfg.signature if hil_runtime is not None else None,
                    "firmware": args.firmware,
                    "control_rate_hz": args.rate,
                    "leader": {
                        "interface": args.hg_leader_interface,
                        "channel": str(args.hg_leader_channel),
                        "role": "same_policy_action_then_zero_force_expert",
                    },
                    "follower": {
                        "interface": args.interface,
                        "channel": str(args.channel),
                        "role": "policy_or_expert_move_js",
                    },
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "policy_type": policy_type,
                    "action_source": "safety_limited_executed_action_full_trial",
                    "observation_source": "follower_feedback",
                    "gate": "manual_keyboard_i_then_relative_motion",
                    "clutch": "follower_minus_mapped_leader_at_takeover",
                    "training_export": "expert_mask_only",
                    "control_source_codes": {
                        "novice": 0,
                        "armed": 1,
                        "expert": 2,
                    },
                    "motion_start": {
                        "joint_threshold_deg": args.hg_start_joint_motion_deg,
                        "gripper_threshold_mm": args.hg_start_gripper_motion_mm,
                        "consecutive_samples": args.hg_start_motion_frames,
                    },
                    "joint_signs": list(args.hg_joint_signs),
                    "joint_offsets_deg": list(args.hg_joint_offsets_deg),
                },
                preview=False,
                encoder_queue_size=args.recording_encoder_queue_size,
                extra_sample_fields=HG_DAGGER_SAMPLE_FIELDS + (("hil_base_action",) if hil_runtime is not None else ()),
                cameras=cameras,
                session_prefix="hil_serl" if hil_runtime is not None else "hg_dagger",
            )
            # Pay the ffmpeg process-start cost before an X trial.  The episode
            # timeline remains stopped until the user starts the full trial.
            hg_recorder.prepare_episode()
            print(
                "[HG-DAgger] dual-arm policy control ready; press X to start. The "
                "same safety-limited ACT action controls both arms. Press I "
                "during an active X trial to hold the follower "
                "and switch the leader to zero-force takeover; deliberate leader "
                "motion marks expert frames. Press I again to return to ACT and "
                "leader shadow following.",
                flush=True,
            )
            if args.hg_label_outcome:
                print(
                    "[HG-DAgger] outcome labeling is enabled for future RL replay; "
                    "this run still executes ACT, not an RL policy or learner.",
                    flush=True,
                )
        if args.execute:
            controls = "Keyboard: tap V to start/save recording; tap X to start/stop; "
            if args.hg_dagger:
                controls += "tap I to take over/return; tap D to discard the active trial; "
            if args.hg_label_outcome:
                controls += "after X stop tap 1=success or 0=failure, manually reset object, then R/Y home; "
            controls += (
                "tap R to reset ACT to home; hold "
                f"Y for {args.keyboard_hold_seconds:.1f}s to home and remain "
                "paused; Ctrl+C homes and exits; Alt+S immediately performs "
                "an electronic emergency stop."
            )
            print(controls, flush=True)
            print(
                "Camera window: V record/save; X start/stop; "
                + ("I expert take over/return; " if args.hg_dagger else "")
                + ("D discard active HG trial; " if args.hg_dagger else "")
                + ("1 success; 0 failure; " if args.hg_label_outcome else "")
                + "R reset ACT; hold Y home; S emergency stop; F fullscreen; Esc windowed.",
                flush=True,
            )
        else:
            print(
                "Keyboard: V starts/saves recording; X/Y/R/Alt+S motion controls "
                "are disabled in shadow mode; Ctrl+C exits without moving the arm.",
                flush=True,
            )
        period = 1.0 / args.rate
        # Shadow validation has a fixed wall-clock duration.  Execute mode is
        # an interactive session: each X-start gets its own duration budget,
        # while paused and homing time must not consume that budget.
        shadow_deadline = time.monotonic() + args.duration
        trial_started_at = None
        trial_id = 0
        chunk_id = 0 if async_planned else -1
        reported_plan_id = None
        chunk_generated_at = (
            active_plan["completed_at"] if async_planned else None
        )
        def record_blocking_motion(state_values, action_values, event):
            """Keep leader shadowing and V recordings alive during R/Y homing."""
            # Paired homing owns the leader for the entire homing and closing
            # phases. Only the pre-home gripper release needs shadow commands.
            if hg_expert is not None and event == "reset_gripper_open":
                hg_expert.command_policy_action(list(action_values))
            if not recorder.active:
                return
            now = time.monotonic()
            recorder.capture(
                now,
                cameras.latest(),
                state_values,
                action_values,
                action_values,
                trial_id,
                False,
                True,
                event,
            )

        def finish_hg_intervention() -> None:
            nonlocal hg_intervening, hg_recording, hg_last_sample
            nonlocal hg_started_at, hg_takeover_action, hg_expert_raw
            nonlocal hg_intervention_id
            hg_intervening = False
            hg_recording = False
            hg_last_sample = None
            hg_started_at = None
            hg_takeover_action = None
            hg_expert_raw = None
            hg_intervention_id = -1
            hg_motion_gate.reset()

        def finish_hg_trial(*, complete: bool, prepare_next: bool = False):
            saved = None
            if hg_recorder is not None and hg_recorder.active:
                if complete:
                    saved = hg_recorder.save_episode()
                    if saved is not None and not saved.name.startswith("episode_"):
                        saved = None
                else:
                    hg_recorder.mark_incomplete()
            if hil_runtime is not None:
                hil_runtime.finish_trial(saved if complete else None)
            if prepare_next and hg_recorder is not None:
                hg_recorder.prepare_episode()
            return saved

        next_status_at = 0.0
        last_status_mode = None
        next_warning_at = 0.0
        next_hil_retry_warning_at = 0.0
        hil_observation_deadline = None
        if hil_runtime is not None:
            from .hil_serl.safety import ObservationDeadline
            hil_observation_deadline = ObservationDeadline(args.hil_max_action_age_ms)
        pending_toggle_event = None
        with report.open("w", encoding="utf-8") as log:
            while args.execute or time.monotonic() < shadow_deadline:
                tick = time.monotonic()
                events = keyboard.poll()
                for key in cameras.poll_keys():
                    if key in (ord("x"), ord("X")):
                        events.append(("x", tick))
                    elif key in (ord("y"), ord("Y")):
                        events.append(("y", tick))
                    elif key in (ord("r"), ord("R")):
                        events.append(("r", tick))
                    elif key in (ord("s"), ord("S")):
                        events.append(("alt+s", tick))
                    elif key in (ord("v"), ord("V")):
                        events.append(("v", tick))
                    elif key in (ord("i"), ord("I")):
                        events.append(("i", tick))
                    elif key in (ord("d"), ord("D")):
                        events.append(("d", tick))
                    elif key == ord("1"):
                        events.append(("1", tick))
                    elif key == ord("0"):
                        events.append(("0", tick))
                if args.hg_label_outcome and pending_outcome_episode is not None:
                    outcome_keys = [key for key, _ in events if key in {"1", "0"}]
                    if outcome_keys:
                        from .manual_hil_outcome import save_manual_outcome
                        outcome = "success" if outcome_keys[0] == "1" else "failure"
                        index = save_manual_outcome(
                            hg_recorder.root, pending_outcome_episode, outcome
                        )
                        if hil_runtime is not None:
                            hil_runtime.label(pending_outcome_episode, outcome)
                        print(
                            f"\n[HIL] episode {index} marked {outcome}; "
                            "reset the object and home the arm before X.", flush=True,
                        )
                        pending_outcome_episode = None
                intervention_toggle = key_tracker.intervention_toggle(events)
                discard_requested = key_tracker.trial_discard(events)
                (
                    x_toggle,
                    home_requested,
                    reset_requested,
                    stop_requested,
                    recording_toggle,
                ) = key_tracker.update(events, tick)
                if recording_toggle:
                    try:
                        if recorder.active:
                            recorder.stop()
                        else:
                            recorder.start()
                    except Exception as exc:
                        print(
                            "\n[record] Could not toggle recording: "
                            f"{type(exc).__name__}: {exc}; model control continues",
                            flush=True,
                        )
                for message in recorder.drain_messages():
                    print(f"\n{message}", flush=True)
                trial_timed_out = (
                    args.execute
                    and motion_enabled
                    and not hg_intervening
                    and trial_started_at is not None
                    and tick - trial_started_at >= args.duration
                )
                if trial_timed_out:
                    finish_hg_intervention()
                    finish_hg_trial(complete=False)
                    motion_enabled = False
                    start_pending = False
                    trial_started_at = None
                    if args.hg_label_outcome:
                        trial_reset_required = True
                    active_actions.clear()
                    active_plan = None
                    planner_requested = False
                    command_target = None
                    previous_joint_velocity = None
                    held_act_action = None
                    latched_act_subgoal = None
                    act_subgoal_previous_joint_error = None
                    act_subgoal_previous_gripper_error = None
                    act_hold_steps = 0
                    act_raw_gap_hard_steps = 0
                    if planner is None and callable(reset):
                        reset()
                    print(
                        f"\nTrial {trial_id} reached {args.duration:g}s; "
                        "model control is paused. Hold Y to home or tap X "
                        "to start a new trial.",
                        flush=True,
                    )
                toggle_event = "trial_timeout" if trial_timed_out else pending_toggle_event
                pending_toggle_event = None
                if discard_requested:
                    if not args.hg_dagger:
                        print("\nD ignored: start with --hg-dagger to enable trial discard.")
                    elif hg_recorder is None or not hg_recorder.active:
                        print(
                            "\n[HG-DAgger] D ignored: no active trial to discard.",
                            flush=True,
                        )
                    else:
                        motion_enabled = False
                        assert hg_expert is not None
                        hg_expert.hold_controlled()
                        finish_hg_intervention()
                        hg_recorder.discard_episode()
                        if hil_runtime is not None:
                            hil_runtime.discard()
                        hg_recorder.prepare_episode()
                        start_pending = False
                        trial_started_at = None
                        trial_reset_required = True
                        active_actions.clear()
                        active_plan = None
                        planner_requested = False
                        command_target = None
                        previous_joint_velocity = None
                        held_act_action = None
                        latched_act_subgoal = None
                        act_subgoal_previous_joint_error = None
                        act_subgoal_previous_gripper_error = None
                        act_hold_steps = 0
                        act_raw_gap_hard_steps = 0
                        if planner is None and callable(reset):
                            reset()
                        toggle_event = "trial_discard"
                        print(
                            f"\n[HG-DAgger] Trial {trial_id} discarded; model control "
                            "is paused. Press R or hold Y to home before the next X.",
                            flush=True,
                        )
                if stop_requested:
                    if not args.execute:
                        print("\nAlt+S ignored: electronic stop is disabled without --execute.")
                        continue
                    motion_enabled = False
                    finish_hg_intervention()
                    finish_hg_trial(complete=False)
                    emergency_stopped = True
                    robot.electronic_emergency_stop()
                    if hg_expert is not None:
                        hg_expert.emergency_stop()
                    if planner is not None:
                        planner.close()
                        planner = None
                    print("\nAlt+S: electronic emergency stop sent; program exits.")
                    return 130
                if home_requested or reset_requested:
                    if not args.execute:
                        print(
                            "\nY/R ignored: homing is disabled without --execute."
                        )
                        continue
                    if reset_requested and policy_type != "act":
                        print("\nR reset is available only for ACT execution.")
                        if not home_requested:
                            continue
                    is_act_reset = reset_requested and policy_type == "act"
                    home_label = "R reset" if is_act_reset else "Y hold"
                    if hg_expert is not None:
                        hg_expert.hold_controlled()
                    finish_hg_intervention()
                    finish_hg_trial(complete=False)
                    motion_enabled = False
                    start_pending = False
                    trial_started_at = None
                    active_actions.clear()
                    active_plan = None
                    planner_requested = False
                    held_act_action = None
                    latched_act_subgoal = None
                    act_subgoal_previous_joint_error = None
                    act_subgoal_previous_gripper_error = None
                    act_hold_steps = 0
                    act_raw_gap_hard_steps = 0
                    if planner is None and callable(reset):
                        reset()
                    print(f"\n{home_label} received: pausing model control and homing.")
                    homing_active = True
                    if is_act_reset:
                        reset_act_to_home(
                            robot, robot_config, gripper, args, expected_js_mode,
                            homing_emergency_stop_requested,
                            record_blocking_motion,
                            hg_expert,
                        )
                        arm_prepared = True
                    else:
                        held_gripper_width = min(
                            max(
                                base.wait_gripper_width(
                                    gripper, min(args.feedback_timeout, 0.2)
                                ),
                                0.0,
                            ),
                            args.gripper_max_mm / 1000.0,
                        )
                        home_arm_to_zero(
                            robot, robot_config, gripper, held_gripper_width,
                            args.gripper_force, args.feedback_timeout,
                            args.home_speed_percent, args.home_timeout,
                            homing_emergency_stop_requested,
                            record_blocking_motion,
                            hg_expert,
                        )
                    if policy_type == "act" and not is_act_reset:
                        held_gripper_width = max(
                            0.0,
                            base.wait_gripper_width(gripper, args.feedback_timeout),
                        )
                        prepare_js_control(
                            robot, robot_config, gripper, held_gripper_width,
                            args.gripper_force, args.feedback_timeout,
                            expected_js_mode, ACT_STREAM_SPEED_PERCENT,
                        )
                        arm_prepared = True
                    elif policy_type != "act":
                        arm_prepared = False
                    command_gate = False
                    command_target = None
                    previous_joint_velocity = None
                    # Clear inputs accumulated during the blocking home/JS
                    # transition, including an early X press.
                    if homing_emergency_stop_requested():
                        emergency_stopped = True
                        robot.electronic_emergency_stop()
                        print("\nAlt+S during reset: electronic emergency stop sent.")
                        return 130
                    homing_active = False
                    key_tracker.after_home(time.monotonic())
                    print("Homing complete; model control remains paused. Press X for a new trial.")
                    trial_reset_required = False
                    continue
                if args.execute and x_toggle:
                    if not motion_enabled and (
                        pending_outcome_episode is not None or trial_reset_required
                    ):
                        if pending_outcome_episode is not None:
                            message = (
                                "label the completed trial with 1/0 and home with "
                                "R/Y before X"
                            )
                        else:
                            message = "home with R/Y before X after discarding the trial"
                        print(f"\n[HG-DAgger] {message}.", flush=True)
                        continue
                    motion_enabled = not motion_enabled
                    toggle_event = "start" if motion_enabled else "stop"
                    if motion_enabled:
                        if hil_runtime is not None:
                            hil_runtime.start_trial()
                            # Policy synchronization happens while paused. The
                            # trial and recording clocks start after it ends.
                            tick = time.monotonic()
                        if hg_recorder is not None:
                            hg_recorder.start_episode(started_at=tick)
                        hg_next_intervention_id = 0
                        trial_id += 1
                        trial_started_at = tick
                        previous_joint_velocity = None
                        print(f"\nTrial {trial_id} started.", flush=True)
                    else:
                        if hg_expert is not None:
                            hg_expert.hold_controlled()
                        finish_hg_intervention()
                        saved_hg_episode = finish_hg_trial(complete=True, prepare_next=True)
                        if args.hg_label_outcome and saved_hg_episode is not None:
                            pending_outcome_episode = saved_hg_episode
                            trial_reset_required = True
                            print(
                                "\n[HIL] trial saved. Press 1 for success or 0 for failure; "
                                "reset the object and home the arm before the next X.",
                                flush=True,
                            )
                        trial_started_at = None
                        active_actions.clear()
                        active_plan = None
                        planner_requested = False
                        command_target = None
                        previous_joint_velocity = None
                        print(f"\nTrial {trial_id} paused.", flush=True)
                    held_act_action = None
                    latched_act_subgoal = None
                    act_subgoal_previous_joint_error = None
                    act_subgoal_previous_gripper_error = None
                    act_hold_steps = 0
                    act_raw_gap_hard_steps = 0
                    if async_planned:
                        if motion_enabled:
                            # Do not start from a chunk generated while the arm
                            # was idle. Wait for a plan from this X-triggered
                            # observation before allowing any command.
                            active_actions.clear()
                            active_plan = None
                            planner_requested = False
                            start_pending = True
                            command_target = None
                        else:
                            start_pending = False
                    elif callable(reset):
                        reset()
                elif x_toggle:
                    print("\nX ignored in shadow mode; no motion is authorized.")

                joint_message = base.wait_for_feedback(robot, min(args.feedback_timeout, 0.2))
                if hil_runtime is not None:
                    from .hil_serl.safety import require_fresh_feedback
                    if hil_can_guard is not None:
                        hil_can_guard.check()
                    require_fresh_feedback(joint_message, "follower joints", min(args.feedback_timeout, 0.5))
                joints = [float(value) for value in joint_message.msg]
                validate_joint_feedback(joints, robot_config)
                gripper_width = base.wait_gripper_width(gripper, min(args.feedback_timeout, 0.2))
                if not math.isfinite(gripper_width) or not -0.003 <= gripper_width <= 0.153:
                    raise RuntimeError(
                        f"gripper feedback {gripper_width * 1000:.2f} mm is implausible"
                    )
                state = np.asarray([*joints, gripper_width], dtype=np.float32)
                frames = cameras.latest()
                image_inputs, camera_ages_ms = camera_batch_images(
                    frames, camera_roles, cv2, torch
                )
                camera_age_ms = max(camera_ages_ms.values())
                batch = {
                    "observation.state": torch.from_numpy(state).unsqueeze(0),
                    **image_inputs,
                }
                add_smolvla_task(batch, policy_type, task)
                if intervention_toggle:
                    if not args.hg_dagger:
                        print("\nI ignored: start with --hg-dagger to enable expert takeover.")
                    elif not motion_enabled:
                        print("\n[HG-DAgger] I ignored while model control is paused; press X first.")
                    elif hg_intervening:
                        print(
                            "\n[HG-DAgger] I received: leaving expert control; "
                            "restoring leader enable and ACT control...",
                            flush=True,
                        )
                        completed_recovery = hg_recording
                        if trial_started_at is not None and hg_started_at is not None:
                            # The duration budget measures novice/model control;
                            # never cut off an expert in the middle of recovery.
                            trial_started_at += tick - hg_started_at
                        assert hg_expert is not None
                        hg_expert.resume_policy_control(state.tolist())
                        finish_hg_intervention()
                        active_actions.clear()
                        active_plan = None
                        planner_requested = False
                        start_pending = async_planned
                        command_target = state.copy()
                        previous_joint_velocity = None
                        held_act_action = None
                        latched_act_subgoal = None
                        act_hold_steps = 0
                        act_raw_gap_hard_steps = 0
                        if planner is None and callable(reset):
                            reset()
                        toggle_event = "hg_return_to_policy"
                        if completed_recovery:
                            print(
                                "\n[HG-DAgger] expert recovery marked in the "
                                "full trial; control returned to a freshly "
                                "reset policy.",
                                flush=True,
                            )
                        else:
                            print(
                                "\n[HG-DAgger] armed takeover cancelled before "
                                "leader motion; the full trial continues.",
                                flush=True,
                            )
                        # Leaving zero-force mode blocks during role/enable
                        # changes. Read new feedback and camera frames next
                        # tick before planning or commanding either arm.
                        command_target = None
                        act_subgoal_previous_joint_error = None
                        act_subgoal_previous_gripper_error = None
                        pending_toggle_event = "hg_return_to_policy"
                        continue
                    else:
                        assert hg_expert is not None and hg_recorder is not None
                        hg_last_sample = engage_expert_with_follower_hold(
                            robot, gripper, hg_expert, state, joint_bounds,
                            args.gripper_force, args.gripper_max_mm / 1000.0,
                        )
                        hg_intervening = True
                        hg_recording = False
                        hg_started_at = tick
                        hg_takeover_action = np.asarray(
                            hg_last_sample.requested_action, dtype=np.float64
                        )
                        hg_motion_gate.arm(hg_last_sample.requested_action)
                        hg_intervention_id = hg_next_intervention_id
                        hg_next_intervention_id += 1
                        active_actions.clear()
                        active_plan = None
                        planner_requested = False
                        start_pending = False
                        command_target = state.copy()
                        previous_joint_velocity = None
                        held_act_action = None
                        latched_act_subgoal = None
                        act_hold_steps = 0
                        act_raw_gap_hard_steps = 0
                        if planner is None and callable(reset):
                            reset()
                        toggle_event = "hg_takeover"
                        print(
                            "\n[HG-DAgger] takeover armed; follower is holding. "
                            "Leader is now in zero-force mode; expert motion and "
                            "labeling start after deliberate leader movement. "
                            "Press I to cancel/return to ACT and leader shadowing.",
                            flush=True,
                        )
                        # engage() blocks while changing gripper drive / arm
                        # role. Do not infer or command from the pre-transition
                        # camera batch. Both I directions resume on fresh data.
                        pending_toggle_event = "hg_takeover"
                        continue
                planner_underflow = False
                plan_observation_age_ms = None
                chunk_inference_ms = None
                act_action_held = False
                act_hold_timeout = False
                act_raw_action_ok = True
                act_raw_action_within_envelope = True
                act_raw_action_failures = []
                act_raw_action_warnings = []
                act_subgoal_created = False
                act_subgoal_reached = False
                act_subgoal_joint_error = None
                act_subgoal_max_joint_error = None
                act_subgoal_gripper_error = None
                act_subgoal_joint_progress = None
                act_subgoal_gripper_progress = None
                hil_action = None
                hil_inference_ms = 0.0
                if hil_runtime is not None and (motion_enabled or not args.execute):
                    hil_started = time.monotonic()
                    hil_action = hil_runtime.predict(
                        batch, autonomous=not hg_intervening,
                        held_action=held_act_action if not hg_intervening else None,
                    )
                    hil_inference_ms = (time.monotonic() - hil_started) * 1000.0
                if hg_intervening:
                    queue_depth_before = len(active_actions) if planner is not None else action_queue_depth(policy)
                    queue_depth_after = queue_depth_before
                    action_index_in_chunk = None
                    chunk_refresh = False
                    chunk_age_ms = None
                    inference_ms = hil_inference_ms
                    inference_limit_ms = args.max_inference_ms
                    raw = state.copy()
                elif planner is not None:
                    if planner.failure is not None:
                        raise RuntimeError(
                            f"async planner failed: {planner.failure}"
                        )
                    queue_depth_before = len(active_actions)
                    observation_at = (min(tick, *(frame[1] for frame in frames.values()))
                                      if policy_type == "pi0" else tick)
                    if policy_type == "pi0" and active_plan is not None:
                        expired = pi0_expired_steps(active_plan["requested_at"], time.monotonic(), args.rate)
                        while active_actions and config.n_action_steps - len(active_actions) < expired:
                            active_actions.popleft()
                    if start_pending and not planner_requested:
                        pending_plan_request_id = planner.request(batch, observation_at)
                        planner_requested = True
                    if not active_actions:
                        next_plan = planner.take_ready(pending_plan_request_id)
                        if next_plan is not None:
                            active_plan = next_plan
                            if policy_type == "pi0":
                                active_actions, _ = trim_pi0_plan(active_plan["actions"], active_plan,
                                    time.monotonic(), args.rate, config.n_action_steps)
                                if not active_actions:
                                    planner_underflow = True
                            else:
                                active_actions.extend(active_plan["actions"])
                            planner_requested = False
                            start_pending = False
                            chunk_id += 1
                            chunk_generated_at = active_plan["completed_at"]
                        elif not start_pending:
                            planner_underflow = True
                    if (
                        active_actions
                        and not start_pending
                        and not planner_requested
                        and (
                        len(active_actions) <= replan_lead_steps
                        )
                    ):
                        pending_plan_request_id = planner.request(batch, observation_at)
                        planner_requested = True
                    if active_actions:
                        action_index_in_chunk = (
                            config.n_action_steps - len(active_actions)
                        )
                        raw = active_actions.popleft()
                        chunk_refresh = (active_plan["request_id"] != reported_plan_id
                                         if policy_type == "pi0" else action_index_in_chunk == 0)
                        reported_plan_id = active_plan["request_id"]
                        chunk_inference_ms = (
                            active_plan["inference_ms"] if chunk_refresh else None
                        )
                        plan_observation_age_ms = (
                            time.monotonic() - active_plan["requested_at"]
                        ) * 1000.0
                    else:
                        action_index_in_chunk = None
                        chunk_refresh = False
                        raw = state.copy()
                    queue_depth_after = len(active_actions)
                    # The control thread performs no model inference. The full
                    # generation cost is recorded separately on chunk refresh.
                    inference_ms = 0.0
                    inference_limit_ms = args.max_inference_ms
                    chunk_age_ms = (
                        (time.monotonic() - chunk_generated_at) * 1000.0
                        if chunk_generated_at is not None else None
                    )
                else:
                    queue_depth_before = action_queue_depth(policy)
                    chunk_refresh = False
                    if args.execute and not motion_enabled:
                        # Keep camera/feedback/watchdog monitoring alive while
                        # paused, but do not advance ACT's action queue.
                        prediction = None
                        inference_ms = 0.0
                        raw = state.copy()
                    elif hil_runtime is not None:
                        inference_ms = hil_inference_ms
                        raw = hil_action
                    elif (
                        policy_type == "act"
                        and held_act_action is not None
                    ):
                        # Freeze both queued ACT and the temporal-ensemble
                        # policy clock while the final safety limiter is under
                        # pressure.  Advancing select_action() here would pop a
                        # temporal step although the robot has not caught up.
                        prediction = None
                        inference_ms = 0.0
                        raw = held_act_action.copy()
                        act_action_held = True
                    else:
                        started = time.monotonic()
                        with torch.inference_mode():
                            prediction = postprocessor(
                                policy.select_action(preprocessor(batch))
                            )
                        inference_ms = (time.monotonic() - started) * 1000.0
                        raw = prediction.detach().cpu().squeeze(0).numpy()
                    queue_depth_after = action_queue_depth(policy)
                    action_index_in_chunk = None
                    chunk_age_ms = None
                    inference_limit_ms = args.max_inference_ms
                hg_sample = None
                hg_expert_raw = None
                if hg_intervening:
                    assert hg_expert is not None
                    novice_raw = (
                        state.copy()
                        if last_novice_action is None
                        else np.asarray(last_novice_action).copy()
                    )
                    hg_sample = hg_last_sample or hg_expert.sample(joint_bounds)
                    hg_last_sample = None
                    hg_expert_raw = np.asarray(
                        hg_sample.requested_action, dtype=np.float64
                    )
                    if not hg_recording and hg_motion_gate.update(
                        hg_sample.requested_action
                    ):
                        hg_recording = True
                        print(
                            "\n[HG-DAgger] leader motion detected; EXPERT "
                            "CONTROL and expert labeling active. Press I to return.",
                            flush=True,
                        )
                    if hg_recording:
                        raw = hg_expert_raw.copy()
                    else:
                        assert hg_takeover_action is not None
                        raw = np.asarray(hg_takeover_action).copy()
                else:
                    novice_raw = np.asarray(raw).copy()
                    last_novice_action = novice_raw.copy()
                tracking_raw = (
                    act_tracking_action(
                        raw, args.gripper_max_mm / 1000.0,
                        args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                    ) if policy_type in {"act", "pi0"} else raw
                )
                control_raw = (
                    latched_act_subgoal
                    if (
                        policy_type == "act"
                        and not hg_intervening
                        and latched_act_subgoal is not None
                    )
                    else tracking_raw
                )
                limiter_previous = (
                    state if command_target is None else command_target
                )
                limiter_velocity = (
                    np.zeros(6, dtype=np.float64)
                    if previous_joint_velocity is None
                    else previous_joint_velocity
                )
                shaped_action, proposed_joint_velocity = (
                    limit_joint_target_acceleration(
                        control_raw, limiter_previous, limiter_velocity,
                        math.radians(args.max_joint_acceleration_deg_s2),
                        period,
                        math.radians(args.max_joint_step_deg) / period,
                    )
                )
                target = clamp_step(
                    shaped_action, state, math.radians(args.max_joint_step_deg),
                    args.max_gripper_step_mm / 1000.0, args.gripper_max_mm / 1000.0,
                    joint_bounds, command_target,
                    math.radians(args.max_joint_lead_deg),
                    args.max_gripper_lead_mm / 1000.0,
                )
                action_clamped = action_was_clamped(raw, target)
                smoothing_applied = action_was_clamped(
                    control_raw[:6], shaped_action[:6]
                )
                # This flag intentionally excludes normal acceleration shaping;
                # it represents only the final hard safety clamp.
                tracking_clamped = action_was_clamped(shaped_action, target)
                backpressure_clamped = (
                    act_backpressure_required(shaped_action, target, state)
                    if policy_type == "act" else tracking_clamped
                )
                if policy_type in {"act", "pi0"} and not hg_intervening:
                    if policy_type == "act":
                        # Only malformed/non-finite actions and physical bounds
                        # are immediate raw-action failures.  A moving raw
                        # target may legitimately outrun feedback, so joint and
                        # gripper gaps are telemetry while a fixed, already
                        # clamped subgoal applies backpressure to execution.
                        immediate_failures = act_action_envelope_failures(
                            raw, state, joint_bounds,
                            args.gripper_max_mm / 1000.0,
                            math.inf,
                            math.inf,
                            args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                            math.radians(
                                args.max_act_raw_joint_bound_tolerance_deg
                            ),
                        )
                        strict_bound_failures = act_action_envelope_failures(
                            raw, state, joint_bounds,
                            args.gripper_max_mm / 1000.0,
                            math.inf,
                            math.inf,
                            args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                            0.0,
                        )
                        if strict_bound_failures and not immediate_failures:
                            act_raw_action_warnings.append(
                                strict_bound_failures[0]
                                + "; within raw prediction tolerance; executable "
                                "target is hard-clamped to the safe joint bounds"
                            )
                        warning_threshold = (
                            args.warn_act_raw_joint_gap_deg
                            if args.warn_act_raw_joint_gap_deg is not None
                            else args.max_act_raw_joint_gap_deg
                        )
                        joint_gap_warning = act_joint_gap_failure(
                            raw, state, math.radians(warning_threshold)
                        )
                        joint_gap_hard = act_joint_gap_failure(
                            raw, state,
                            math.radians(args.max_act_raw_joint_gap_deg),
                        )
                        (
                            act_raw_gap_hard_steps,
                            _persistent_joint_gap_failure,
                        ) = update_act_raw_gap_streak(
                            joint_gap_hard,
                            act_raw_gap_hard_steps,
                            args.max_act_raw_gap_consecutive_steps,
                        )
                        if joint_gap_warning is not None:
                            warning = (
                                joint_gap_warning
                                + "; raw telemetry only; ACT action consumption "
                                "is paced by safety backpressure"
                            )
                            if joint_gap_hard is not None:
                                warning += (
                                    "; severe gap for "
                                    f"{act_raw_gap_hard_steps} consecutive ticks"
                                )
                            act_raw_action_warnings.append(warning)
                        gripper_gap = abs(float(raw[6] - state[6]))
                        if gripper_gap > (
                            args.max_act_raw_gripper_gap_mm / 1000.0
                        ):
                            act_raw_action_warnings.append(
                                "raw gripper gap="
                                f"{gripper_gap * 1000.0:.2f}mm > "
                                f"{args.max_act_raw_gripper_gap_mm:.2f}mm; "
                                "raw telemetry only; ACT action consumption is "
                                "paced by safety backpressure"
                            )
                        act_raw_action_failures = list(immediate_failures)
                        act_raw_action_within_envelope = not (
                            immediate_failures
                            or joint_gap_warning is not None
                            or gripper_gap
                            > args.max_act_raw_gripper_gap_mm / 1000.0
                        )
                    else:
                        act_raw_action_failures = act_action_envelope_failures(
                            raw, state, joint_bounds,
                            args.gripper_max_mm / 1000.0,
                            math.radians(args.max_act_raw_joint_gap_deg),
                            args.max_act_raw_gripper_gap_mm / 1000.0,
                            args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                        )
                        act_raw_gap_hard_steps = 0
                        act_raw_action_within_envelope = not act_raw_action_failures
                    act_raw_action_ok = not act_raw_action_failures
                    if policy_type == "act" and args.execute and motion_enabled:
                        candidate_subgoal = bounded_act_subgoal(
                            tracking_raw,
                            state,
                            joint_bounds,
                            proposed_joint_velocity,
                            math.radians(args.max_joint_lead_deg),
                            args.max_gripper_lead_mm / 1000.0,
                            args.gripper_max_mm / 1000.0,
                            math.radians(args.max_joint_acceleration_deg_s2),
                        )
                        (
                            latched_act_subgoal,
                            act_subgoal_created,
                            act_subgoal_reached,
                        ) = update_act_backpressure_target(
                            latched_act_subgoal,
                            candidate_subgoal,
                            backpressure_clamped,
                            act_raw_action_ok,
                        )
                        if act_subgoal_created:
                            # Freeze ACT until joint/opening backpressure ends;
                            # closing width alone must not freeze a grasp.
                            (
                                act_subgoal_joint_error,
                                act_subgoal_max_joint_error,
                                act_subgoal_gripper_error,
                            ) = act_subgoal_errors(latched_act_subgoal, state)
                            act_subgoal_previous_joint_error = (
                                act_subgoal_joint_error
                            )
                            act_subgoal_previous_gripper_error = (
                                act_subgoal_gripper_error
                            )
                            act_hold_steps = 0
                            held_act_action = latched_act_subgoal.copy()
                        elif act_subgoal_reached:
                            # The command is no longer being hard-clamped.  Do
                            # not wait for Piper JS to settle at an exact point;
                            # resume ACT from the next observation/time step.
                            held_act_action = None
                            act_subgoal_previous_joint_error = None
                            act_subgoal_previous_gripper_error = None
                            act_hold_steps = 0
                        elif latched_act_subgoal is not None:
                            held_act_action = latched_act_subgoal.copy()
                            subgoal_progress = update_act_backpressure_progress(
                                latched_act_subgoal,
                                state,
                                act_subgoal_previous_joint_error,
                                act_subgoal_previous_gripper_error,
                                act_hold_steps,
                                math.radians(args.act_subgoal_min_progress_deg),
                                args.act_subgoal_min_gripper_progress_mm / 1000.0,
                            )
                            act_subgoal_joint_error = subgoal_progress[
                                "joint_error"
                            ]
                            act_subgoal_max_joint_error = subgoal_progress[
                                "max_joint_error"
                            ]
                            act_subgoal_gripper_error = subgoal_progress[
                                "gripper_error"
                            ]
                            act_subgoal_joint_progress = subgoal_progress[
                                "joint_progress"
                            ]
                            act_subgoal_gripper_progress = subgoal_progress[
                                "gripper_progress"
                            ]
                            act_hold_steps = subgoal_progress[
                                "no_progress_steps"
                            ]
                            # Keep the last acknowledged baseline so small
                            # feedback increments accumulate across ticks.
                            if subgoal_progress["progressed"]:
                                act_subgoal_previous_joint_error = (
                                    act_subgoal_joint_error
                                )
                                act_subgoal_previous_gripper_error = (
                                    act_subgoal_gripper_error
                                )
                            act_hold_timeout = (
                                act_hold_steps >= args.max_act_clamp_hold_steps
                            )
                        else:
                            held_act_action = None
                            act_hold_steps = 0
                    else:
                        held_act_action = None
                        latched_act_subgoal = None
                        act_subgoal_previous_joint_error = None
                        act_subgoal_previous_gripper_error = None
                        act_hold_steps = 0
                execution_ready = transition_is_safe(
                    joints, target[:6], joint_bounds
                )
                mode_feedback = read_mode_feedback(robot)
                hil_action_age_ms = None
                if hil_runtime is not None:
                    # Acquisition freshness (50 ms by default) and full
                    # observation-to-command age have different budgets.
                    if hil_action is not None:
                        hil_action_age_ms = max((time.monotonic() - frames[role][1]) * 1000 for role in camera_roles)
                    if hil_can_guard is not None:
                        hil_can_guard.check()
                    require_fresh_feedback(joint_message, "follower joints", min(args.feedback_timeout, 0.5))
                    require_fresh_feedback(gripper.get_gripper_status(), "follower gripper", min(args.feedback_timeout, 0.5))
                mode_ready = not arm_prepared or mode_feedback == expected_js_mode
                control_checks_healthy = (
                    camera_age_ms <= args.max_camera_age_ms
                    and inference_ms <= inference_limit_ms
                    and execution_ready
                    and mode_ready
                    and not planner_underflow
                    and (policy_type != "pi0" or act_raw_action_ok)
                    and (
                        policy_type != "act"
                        or not args.execute
                        or not motion_enabled
                        or (act_raw_action_ok and not act_hold_timeout)
                    )
                    and (
                        not async_planned
                        or (
                            chunk_inference_ms is None
                            or chunk_inference_ms <= max_chunk_inference_ms
                        )
                    )
                    and (
                        plan_observation_age_ms is None
                        or plan_observation_age_ms
                        <= args.max_smolvla_plan_age_ms
                    )
                )
                healthy = control_checks_healthy and (
                    hil_action_age_ms is None
                    or hil_action_age_ms <= args.hil_max_action_age_ms
                )
                performance_warnings, inference_failures = inference_timing_messages(
                    inference_ms,
                    args.warn_inference_ms,
                    inference_limit_ms,
                )
                performance_warnings.extend(act_raw_action_warnings)
                watchdog_failures = []
                if camera_age_ms > args.max_camera_age_ms:
                    watchdog_failures.append(
                        f"camera age {camera_age_ms:.1f}ms > "
                        f"{args.max_camera_age_ms:.1f}ms"
                    )
                if hil_action_age_ms is not None and hil_action_age_ms > args.hil_max_action_age_ms:
                    watchdog_failures.append(
                        f"HIL observation-to-command age {hil_action_age_ms:.1f}ms > "
                        f"{args.hil_max_action_age_ms:.1f}ms"
                    )
                watchdog_failures.extend(inference_failures)
                if not execution_ready:
                    watchdog_failures.append("safe target transition is outside joint bounds")
                if not mode_ready:
                    watchdog_failures.append(
                        f"Piper mode feedback {mode_feedback} != {expected_js_mode}"
                    )
                if planner_underflow:
                    watchdog_failures.append("async planner action queue underflow")
                if policy_type == "pi0" or (policy_type == "act" and args.execute and motion_enabled):
                    watchdog_failures.extend(act_raw_action_failures)
                    if act_hold_timeout:
                        watchdog_failures.append(
                            "ACT hard-clamped backpressure target made no "
                            "feedback progress for "
                            f"{act_hold_steps} consecutive ticks >= "
                            f"{args.max_act_clamp_hold_steps}"
                        )
                if (
                    chunk_inference_ms is not None
                    and chunk_inference_ms > max_chunk_inference_ms
                ):
                    watchdog_failures.append(
                        f"chunk inference {chunk_inference_ms:.1f}ms > "
                        f"{max_chunk_inference_ms:.1f}ms"
                    )
                if (
                    plan_observation_age_ms is not None
                    and plan_observation_age_ms > args.max_smolvla_plan_age_ms
                ):
                    watchdog_failures.append(
                        f"plan observation age {plan_observation_age_ms:.1f}ms > "
                        f"{args.max_smolvla_plan_age_ms:.1f}ms"
                    )
                retry_hil_observation = (
                    hil_observation_deadline is not None
                    and hil_observation_deadline.retry(
                        hil_action_age_ms,
                        other_checks_healthy=(
                            args.execute and motion_enabled and not start_pending
                            and control_checks_healthy and len(watchdog_failures) == 1
                        ),
                    )
                )
                if retry_hil_observation:
                    # Only the observation deadline failed. Discard the entire
                    # prediction, including any subgoal created from it. Keep
                    # the last bounded command while fetching fresh data; never
                    # replay this late action. The third consecutive miss stops
                    # the trial via the normal, deliberately latched watchdog.
                    hil_runtime.discard_observation()
                    previous_joint_velocity = None
                    held_act_action = None
                    latched_act_subgoal = None
                    act_subgoal_previous_joint_error = None
                    act_subgoal_previous_gripper_error = None
                    act_hold_steps = 0
                    log.write(json.dumps({
                        "event": "hil_observation_frame_skipped",
                        "trial_id": trial_id,
                        "hg_intervening": bool(hg_intervening),
                        "hg_recording": bool(hg_recording),
                        "inference_ms": hil_inference_ms,
                        "camera_age_ms": camera_age_ms,
                        "hil_action_age_ms": hil_action_age_ms,
                        "hil_consecutive_deadline_misses": hil_observation_deadline.misses,
                        "command_sent": False,
                    }) + "\n")
                    log.flush()
                    if time.monotonic() >= next_hil_retry_warning_at:
                        next_hil_retry_warning_at = time.monotonic() + 1.0
                        print(
                            f"\n[HIL] slow observation "
                            f"({hil_action_age_ms:.1f}ms); frame skipped, "
                            "last targets retained; retrying with fresh data "
                            f"({hil_observation_deadline.misses}/"
                            f"{hil_observation_deadline.maximum_consecutive_misses}).",
                            flush=True,
                        )
                    continue
                if (hil_observation_deadline is not None
                        and hil_observation_deadline.misses
                        >= hil_observation_deadline.maximum_consecutive_misses):
                    watchdog_failures.append(
                        f"HIL observation deadline missed for "
                        f"{hil_observation_deadline.misses} consecutive frames"
                    )
                # A watchdog trip always disarms the toggle. Recovery requires
                # another deliberate X press; motion must never restart itself.
                if (
                    args.execute and motion_enabled and not healthy
                    and not start_pending
                ):
                    motion_enabled = False
                    if hg_expert is not None:
                        hg_expert.hold_controlled()
                    finish_hg_intervention()
                    finish_hg_trial(complete=False)
                    trial_started_at = None
                    toggle_event = "watchdog_stop"
                    held_act_action = None
                    latched_act_subgoal = None
                    act_subgoal_previous_joint_error = None
                    act_subgoal_previous_gripper_error = None
                    previous_joint_velocity = None
                    if planner is None and callable(reset):
                        reset()
                    detail = "; ".join(watchdog_failures) or "unknown safety condition"
                    print(
                        f"\nWATCHDOG: Trial {trial_id} paused before the next "
                        f"command: {detail}",
                        flush=True,
                    )
                elif (args.execute and motion_enabled and performance_warnings
                      and tick >= next_warning_at):
                    next_warning_at = tick + 1.0
                    print(
                        "\nWARNING: " + "; ".join(performance_warnings)
                        + "; continuing execution.",
                        flush=True,
                    )
                may_command = (
                    args.execute and healthy and motion_enabled
                    and not start_pending
                )
                if may_command and not arm_prepared:
                    prepare_js_control(
                        robot, robot_config, gripper, gripper_width,
                        args.gripper_force, args.feedback_timeout,
                        expected_js_mode,
                        ACT_STREAM_SPEED_PERCENT if policy_type == "act" else None,
                    )
                    arm_prepared = True
                    mode_feedback = expected_js_mode
                    if policy_type == "pi0":
                        # Mode setup may block for hundreds of ms. Revalidate
                        # timestamp and feedback next tick, never send that old action.
                        may_command = False
                if may_command and not command_gate:
                    command_gate = True
                command_joint_velocity = None
                command_joint_acceleration = None
                if may_command:
                    if hg_expert is not None and not hg_intervening:
                        hg_expert.command_policy_action(target.tolist())
                    robot.move_js(target[:6].tolist())
                    gripper.move_gripper_m(float(target[6]), args.gripper_force)
                    command_joint_velocity = (
                        np.asarray(target[:6]) - np.asarray(limiter_previous[:6])
                    ) / period
                    command_joint_acceleration = (
                        command_joint_velocity - limiter_velocity
                    ) / period
                    previous_joint_velocity = command_joint_velocity.copy()
                    command_target = target.copy()
                elif command_gate:
                    robot.move_js(joints)
                    gripper.move_gripper_m(gripper_width, args.gripper_force)
                    command_gate = False
                    command_target = state.copy()
                    previous_joint_velocity = None
                elif args.execute:
                    command_target = state.copy()
                    previous_joint_velocity = None

                if hg_recorder is not None and hg_recorder.active:
                    control_source = 2 if hg_recording else (1 if hg_intervening else 0)
                    executed_action = target if may_command else state
                    leader_state = (
                        hg_sample.leader_state
                        if hg_sample is not None
                        else [math.nan] * 7
                    )
                    expert_requested = (
                        hg_expert_raw.tolist()
                        if hg_expert_raw is not None
                        else [math.nan] * 7
                    )
                    leader_age = (
                        hg_sample.feedback_age
                        if hg_sample is not None
                        else math.nan
                    )
                    hg_recorder.capture(
                        tick,
                        state.tolist(),
                        np.asarray(executed_action).tolist(),
                        list(robot.fk(joints)),
                        (
                            np.asarray(executed_action) - np.asarray(state)
                        )[:6].tolist(),
                        {
                            "leader_state": leader_state,
                            "novice_action": np.asarray(novice_raw).tolist(),
                            "expert_requested_action": expert_requested,
                            "following_error": (
                                np.asarray(executed_action) - np.asarray(state)
                            ).tolist(),
                            "leader_feedback_age": leader_age,
                            "control_source": control_source,
                            "expert_mask": int(hg_recording),
                            "intervention_id": hg_intervention_id,
                            "command_sent": int(may_command),
                            **({"hil_base_action": hil_runtime.base_action.tolist()}
                               if hil_runtime is not None else {}),
                        },
                        show_preview=False,
                        frames=frames,
                    )
                    if hil_runtime is not None:
                        hil_runtime.capture(executed_action, expert=hg_recording, command_sent=may_command)

                recorder.capture(
                    tick,
                    frames,
                    state,
                    target if may_command else state,
                    raw,
                    trial_id,
                    bool(args.execute and motion_enabled),
                    bool(may_command),
                    toggle_event,
                )
                cameras.show(
                    frames,
                    recorder.active
                    or bool(hg_recorder is not None and hg_recorder.active),
                )
                control_loop_ms = (time.monotonic() - tick) * 1000.0
                control_deadline_missed = control_loop_ms > period * 1000.0

                row = {
                    "time": time.time(), "checkpoint_sha256": digest,
                    "policy_type": policy_type,
                    "hil_learner_step": hil_runtime.version if hil_runtime is not None else None,
                    "hil_action_age_ms": hil_action_age_ms,
                    "hil_consecutive_deadline_misses": (
                        hil_observation_deadline.misses
                        if hil_observation_deadline is not None else None
                    ),
                    "hil_max_action_age_ms": args.hil_max_action_age_ms if hil_runtime is not None else None,
                    "hil_base_action": (hil_runtime.base_action.tolist()
                                        if hil_runtime is not None and hil_runtime.base_action is not None else None),
                    **({"pi0_clock_aligned": True,
                        "base_checkpoint": str(pi0_base)} if policy_type == "pi0" else {}),
                    "mode": "execute" if args.execute else "shadow",
                    "camera_age_ms": camera_age_ms, "inference_ms": inference_ms,
                    "inference_limit_ms": inference_limit_ms,
                    "rate_hz": args.rate,
                    "warn_inference_ms": args.warn_inference_ms,
                    "max_inference_ms": args.max_inference_ms,
                    "max_chunk_inference_ms": max_chunk_inference_ms,
                    "planner_mode": (
                        "async_chunked" if async_planned else None
                    ),
                    "chunk_inference_ms": chunk_inference_ms,
                    "plan_observation_age_ms": plan_observation_age_ms,
                    "max_smolvla_plan_age_ms": args.max_smolvla_plan_age_ms,
                    "smolvla_replan_lead_steps": replan_lead_steps,
                    "planner_underflow": planner_underflow,
                    "max_camera_age_ms": args.max_camera_age_ms,
                    "device": args.device,
                    "interface": args.interface,
                    "channel": str(args.channel),
                    "firmware": args.firmware,
                    "camera_serial": args.camera_serial,
                    **({"camera_serials": camera_serials} if "wrist" in camera_roles else {}),
                    "camera_ages_ms": camera_ages_ms,
                    "camera_roles": camera_roles,
                    "camera_fps": args.camera_fps,
                    "max_joint_step_deg": args.max_joint_step_deg,
                    "max_joint_acceleration_deg_s2": (
                        args.max_joint_acceleration_deg_s2
                    ),
                    "max_gripper_step_mm": args.max_gripper_step_mm,
                    "max_joint_lead_deg": args.max_joint_lead_deg,
                    "max_gripper_lead_mm": args.max_gripper_lead_mm,
                    "warn_act_raw_joint_gap_deg": (
                        args.warn_act_raw_joint_gap_deg
                    ),
                    "max_act_raw_joint_gap_deg": args.max_act_raw_joint_gap_deg,
                    "max_act_raw_gap_consecutive_steps": (
                        args.max_act_raw_gap_consecutive_steps
                    ),
                    "max_act_raw_gripper_gap_mm": args.max_act_raw_gripper_gap_mm,
                    "max_act_raw_joint_bound_tolerance_deg": (
                        args.max_act_raw_joint_bound_tolerance_deg
                    ),
                    "max_act_raw_gripper_bound_tolerance_mm": (
                        args.max_act_raw_gripper_bound_tolerance_mm
                    ),
                    "max_act_clamp_hold_steps": args.max_act_clamp_hold_steps,
                    "act_subgoal_min_progress_deg": (
                        args.act_subgoal_min_progress_deg
                    ),
                    "act_subgoal_min_gripper_progress_mm": (
                        args.act_subgoal_min_gripper_progress_mm
                    ),
                    "gripper_max_mm": args.gripper_max_mm,
                    "gripper_force": args.gripper_force,
                    "control_period_ms": period * 1000.0,
                    "control_loop_ms": control_loop_ms,
                    "control_deadline_missed": control_deadline_missed,
                    "healthy": healthy,
                    "control_gate": "keyboard_x_toggle",
                    "x_pressed": x_toggle,
                    "keyboard_x_event": x_toggle,
                    "motion_enabled": motion_enabled,
                    "trial_id": trial_id if args.execute else None,
                    "trial_elapsed_s": (
                        tick - trial_started_at
                        if trial_started_at is not None else None
                    ),
                    "trial_timed_out": trial_timed_out,
                    "toggle_event": toggle_event, "command_sent": may_command,
                    "recording_active": recorder.active,
                    "hg_dagger_enabled": bool(args.hg_dagger),
                    "hg_intervening": bool(hg_intervening),
                    "hg_recording": bool(hg_recording),
                    "hg_motion_streak": hg_motion_gate.streak,
                    "hg_expert_action_recorded": bool(
                        hg_recording
                        and hg_recorder is not None
                        and hg_recorder.active
                        and may_command
                    ),
                    "hg_dataset_root": (
                        None
                        if hg_recorder is None or hg_recorder.root is None
                        else str(hg_recorder.root)
                    ),
                    "recording_root": (
                        None
                        if recorder.session_root is None
                        else str(recorder.session_root)
                    ),
                    "n_action_steps": getattr(config, "n_action_steps", None),
                    **({
                        "chunk_size": getattr(config, "chunk_size", None),
                        "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
                    } if policy_type in {"act", "pi0"} else {}),
                    "task": task,
                    "num_flow_steps": getattr(config, "num_steps", None),
                    "chunk_refresh": chunk_refresh,
                    "chunk_id": chunk_id if async_planned else None,
                    "action_index_in_chunk": action_index_in_chunk,
                    "action_queue_depth_before": queue_depth_before,
                    "action_queue_depth_after": queue_depth_after,
                    "action_clamped": action_clamped,
                    "joint_smoothing_applied": smoothing_applied,
                    "shaped_action": np.asarray(shaped_action).tolist(),
                    "proposed_joint_velocity_deg_s": np.degrees(
                        proposed_joint_velocity
                    ).tolist(),
                    "command_joint_velocity_deg_s": (
                        None if command_joint_velocity is None
                        else np.degrees(command_joint_velocity).tolist()
                    ),
                    "command_joint_acceleration_deg_s2": (
                        None if command_joint_acceleration is None
                        else np.degrees(command_joint_acceleration).tolist()
                    ),
                    **({
                        "act_tracking_action": np.asarray(tracking_raw).tolist(),
                        "act_control_action": np.asarray(control_raw).tolist(),
                        "act_tracking_clamped": tracking_clamped,
                        "act_backpressure_clamped": backpressure_clamped,
                        "act_gripper_boundary_corrected": bool(raw[6] != tracking_raw[6]),
                    } if policy_type in {"act", "pi0"} else {}),
                    "act_action_held": act_action_held,
                    "act_policy_clock_frozen": act_action_held,
                    "act_hold_steps": act_hold_steps,
                    "act_hold_timeout": act_hold_timeout,
                    "act_backpressure_active": latched_act_subgoal is not None,
                    "act_backpressure_created": act_subgoal_created,
                    "act_backpressure_released": act_subgoal_reached,
                    "act_backpressure_no_progress_steps": act_hold_steps,
                    "act_subgoal_active": latched_act_subgoal is not None,
                    "act_subgoal_created": act_subgoal_created,
                    "act_subgoal_reached": act_subgoal_reached,
                    "act_subgoal": (
                        None if latched_act_subgoal is None
                        else np.asarray(latched_act_subgoal).tolist()
                    ),
                    "act_subgoal_joint_error_deg": (
                        None if act_subgoal_joint_error is None
                        else math.degrees(act_subgoal_joint_error)
                    ),
                    "act_subgoal_max_joint_error_deg": (
                        None if act_subgoal_max_joint_error is None
                        else math.degrees(act_subgoal_max_joint_error)
                    ),
                    "act_subgoal_gripper_error_mm": (
                        None if act_subgoal_gripper_error is None
                        else act_subgoal_gripper_error * 1000.0
                    ),
                    "act_subgoal_joint_progress_deg": (
                        None if act_subgoal_joint_progress is None
                        else math.degrees(act_subgoal_joint_progress)
                    ),
                    "act_subgoal_gripper_progress_mm": (
                        None if act_subgoal_gripper_progress is None
                        else act_subgoal_gripper_progress * 1000.0
                    ),
                    "act_subgoal_no_progress_steps": act_hold_steps,
                    "act_raw_action_ok": act_raw_action_ok,
                    "act_raw_action_within_envelope": (
                        act_raw_action_within_envelope
                    ),
                    "act_raw_gap_hard_steps": act_raw_gap_hard_steps,
                    "act_raw_action_warnings": act_raw_action_warnings,
                    "watchdog_failures": watchdog_failures,
                    "performance_warnings": performance_warnings,
                    "chunk_age_ms": chunk_age_ms,
                    "num_inference_steps": getattr(
                        config, "num_inference_steps", None
                    ),
                    "temporal_ensemble_coeff": getattr(
                        config, "temporal_ensemble_coeff", None
                    ),
                    "execution_ready": execution_ready,
                    "arm_prepared": arm_prepared,
                    "mode_feedback": mode_feedback,
                    "state": state.tolist(), "raw_action": raw.tolist(), "safe_action": target.tolist(),
                    "hg_expert_requested_action": (
                        None
                        if hg_expert_raw is None
                        else np.asarray(hg_expert_raw).tolist()
                    ),
                    "novice_action_before_or_during_intervention": (
                        np.asarray(novice_raw).tolist()
                        if hg_intervening else None
                    ),
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if hg_recording and may_command:
                    mode = "EXPERT"
                elif hg_intervening and may_command:
                    mode = "ARMED"
                elif not healthy:
                    mode = "WATCHDOG"
                elif may_command:
                    mode = "EXEC"
                else:
                    mode = "READY" if args.execute else "SHADOW"
                chunk_label = " refresh" if chunk_refresh else ""
                if mode != last_status_mode or tick >= next_status_at:
                    print(
                        f"\r{mode}{chunk_label} infer={inference_ms:5.1f}ms "
                        f"loop={control_loop_ms:5.1f}ms "
                        f"camera={camera_age_ms:5.1f}ms"
                        + (
                            f" plan_age={plan_observation_age_ms:5.1f}ms"
                            if plan_observation_age_ms is not None else ""
                        ),
                        end="", flush=True,
                    )
                    last_status_mode = mode
                    next_status_at = tick + 1.0
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        print(f"\nReport: {report}")
        return 0
    except KeyboardInterrupt:
        if robot is not None and args.execute:
            motion_enabled = False
            if homing_active:
                emergency_stopped = True
                robot.electronic_emergency_stop()
                if hg_expert is not None:
                    hg_expert.emergency_stop()
                print("\nCtrl+C during homing: electronic emergency stop sent.")
                return 130
            print("\nCtrl+C received: pausing model control, homing, then exiting.")
            homing_active = True
            try:
                if hg_expert is not None:
                    hg_expert.hold_controlled()
                held_gripper_width = min(
                    max(
                        base.wait_gripper_width(
                            gripper, min(args.feedback_timeout, 0.2)
                        ),
                        0.0,
                    ),
                    args.gripper_max_mm / 1000.0,
                )
                home_arm_to_zero(
                    robot, robot_config, gripper, held_gripper_width,
                    args.gripper_force, args.feedback_timeout,
                    args.home_speed_percent, args.home_timeout,
                    homing_emergency_stop_requested,
                    record_blocking_motion,
                    hg_expert,
                )
                homing_active = False
                closed_width = base.wait_gripper_width(gripper, args.feedback_timeout)
                prepare_js_control(
                    robot, robot_config, gripper, closed_width,
                    args.gripper_force, args.feedback_timeout, expected_js_mode,
                    ACT_STREAM_SPEED_PERCENT if policy_type == "act" else None,
                )
                if hg_expert is not None:
                    hg_expert.hold_controlled()
                arm_prepared = True
                print("Ctrl+C: homing complete; both arms remain enabled and holding; program exits.")
                return 0
            except KeyboardInterrupt:
                emergency_stopped = True
                robot.electronic_emergency_stop()
                if hg_expert is not None:
                    hg_expert.emergency_stop()
                print("\nSecond Ctrl+C during homing: electronic emergency stop sent.")
                return 130
            except LeaderHomingFailure as exc:
                follower_error, leader_error = isolate_leader_home_failure(
                    robot, robot_config, args.feedback_timeout, hg_expert
                )
                homing_active = False
                if follower_error is None:
                    print(
                        "\nCtrl+C homing stopped: leader E-stopped; follower "
                        "is holding its measured position."
                    )
                else:
                    emergency_stopped = True
                    print(
                        "\nCtrl+C homing stopped: leader E-stopped; follower "
                        "hold failed, so follower was also E-stopped: "
                        f"{type(follower_error).__name__}: {follower_error}"
                    )
                if leader_error is not None:
                    print(
                        "WARNING: leader E-stop command failed: "
                        f"{type(leader_error).__name__}: {leader_error}"
                    )
                print(f"Leader homing failure: {exc}")
                return 1
            except Exception as exc:
                emergency_stopped = True
                robot.electronic_emergency_stop()
                if hg_expert is not None:
                    hg_expert.emergency_stop()
                print(
                    "\nCtrl+C homing failed; electronic emergency stop sent: "
                    f"{type(exc).__name__}: {exc}"
                )
                return 1
        print("\nCtrl+C: shadow inference exits without commanding the arm.")
        return 0
    except LeaderHomingFailure as exc:
        if robot is not None and args.execute and hg_expert is not None:
            motion_enabled = False
            follower_error, leader_error = isolate_leader_home_failure(
                robot, robot_config, args.feedback_timeout, hg_expert
            )
            homing_active = False
            if follower_error is None:
                print(
                    "\nLeader homing failed: only the leader was E-stopped; "
                    "the follower is holding its measured position.",
                    flush=True,
                )
            else:
                emergency_stopped = True
                print(
                    "\nLeader homing failed and the follower could not hold; "
                    "both arms were E-stopped: "
                    f"{type(follower_error).__name__}: {follower_error}",
                    flush=True,
                )
            if leader_error is not None:
                print(
                    "WARNING: leader E-stop command failed: "
                    f"{type(leader_error).__name__}: {leader_error}",
                    flush=True,
                )
            print(f"Leader homing failure: {exc}", flush=True)
            return 1
        raise
    except Exception:
        if robot is not None and (homing_active or (args.hil_run and args.execute)):
            emergency_stopped = True
            robot.electronic_emergency_stop()
            if hg_expert is not None:
                hg_expert.emergency_stop()
            print("\nControl failed: electronic emergency stop sent.")
        raise
    finally:
        keyboard.close()
        if hil_runtime is not None:
            hil_runtime.close()
        if planner is not None:
            planner.close()
        if robot is not None and arm_prepared and not emergency_stopped:
            try:
                hold = read_valid_joints(robot, robot_config, 0.5)
                robot.move_js(hold)
            except Exception:
                pass
        # Restore the physical leader hold before waiting for any video
        # encoder queues to drain during recorder shutdown.
        if hg_expert is not None:
            hg_expert.close()
        if recorder is not None:
            recorder.close()
            for message in recorder.drain_messages():
                print(f"\n{message}", flush=True)
        if hg_recorder is not None:
            hg_recorder.close()
        if cameras is not None:
            cameras.close()
        base.disconnect_quietly(robot)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nOnline policy stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
