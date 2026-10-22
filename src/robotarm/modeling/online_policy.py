#!/usr/bin/env python3
"""Piper policy runner; uncalibrated Pi0.5 is restricted to shadow inference."""

from __future__ import annotations

import argparse
from collections import deque
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
from ..paths import ARTIFACTS_ROOT, MODELING_CONFIGS_ROOT, TRAINING_LOG_ROOT, WORKSPACE_ROOT
from ..teleop import keyboard as keyboard_api


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
    expected_js_mode,
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
    time.sleep(0.1)
    joints = read_valid_joints(robot, robot_config, timeout)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
    robot.move_js(joints)
    wait_mode_feedback(robot, expected_js_mode, timeout)
    return joints


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
    )


def act_action_envelope_failures(
    raw, current, joint_bounds, gripper_max,
    max_raw_joint_gap, max_raw_gripper_gap,
    raw_gripper_bound_tolerance=0.0,
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
    for index, (value, (low, high)) in enumerate(
        zip(raw[:6], joint_bounds), start=1
    ):
        if not low <= value <= high:
            outside.append(
                f"J{index}={math.degrees(value):.2f}deg "
                f"outside [{math.degrees(low):.2f}, {math.degrees(high):.2f}]deg"
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

    joint_gaps = np.abs(raw[:6] - current[:6])
    largest_joint = int(np.argmax(joint_gaps))
    if joint_gaps[largest_joint] > max_raw_joint_gap:
        reasons.append(
            f"raw joint gap J{largest_joint + 1}="
            f"{math.degrees(joint_gaps[largest_joint]):.2f}deg > "
            f"{math.degrees(max_raw_joint_gap):.2f}deg"
        )
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
    """Convert keyboard events into X-toggle, Y-hold, and Alt+S stop intents."""

    def __init__(self, hold_seconds, repeat_gap=0.75):
        self.hold_seconds = hold_seconds
        self.repeat_gap = repeat_gap
        self.last_x = None
        self.y_started = None
        self.last_y = None
        self.y_event_count = 0
        self.y_fired = False

    def update(self, events, now):
        x_toggle = False
        emergency_stop = False
        for key, observed_at in events:
            if key == "x":
                if (
                    self.last_x is None
                    or observed_at - self.last_x > self.repeat_gap
                ):
                    x_toggle = True
                self.last_x = observed_at
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
        return x_toggle, y_hold, emergency_stop


def home_arm_to_zero(
    robot, robot_config, gripper, gripper_width, gripper_force,
    feedback_timeout, speed_percent, timeout,
):
    """Move all six joints to zero with MOVE_J and verified feedback."""
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
    robot.move_j(target)

    started = time.monotonic()
    deadline = started + timeout
    tolerance = math.radians(3.0)
    first_error = None
    no_motion_deadline = None
    print(
        f"Homing: MOVE_J to zero at {speed_percent}% speed; "
        "gripper holds its current width.",
        flush=True,
    )
    while time.monotonic() < deadline:
        joints = read_valid_joints(
            robot, robot_config, min(feedback_timeout, 0.2)
        )
        status = robot.get_arm_status()
        max_error = max(abs(value) for value in joints)
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
        print(
            f"\rHOMING joint_error={math.degrees(max_error):6.2f} deg",
            end="", flush=True,
        )
        if (
            motion_done
            and time.monotonic() - started >= 0.5
            and max_error <= tolerance
        ):
            robot.move_j(joints)
            print("\nHoming complete; arm is at zero.")
            return joints
        if (
            time.monotonic() >= no_motion_deadline
            and max_error > first_error - math.radians(0.5)
        ):
            raise RuntimeError(
                "homing command produced no joint motion within 3s "
                "(joint_error stuck at "
                f"{math.degrees(max_error):.2f} deg); the controller likely "
                "dropped the J-mode MOVE_J target"
            )
        time.sleep(0.05)
    raise TimeoutError(
        f"homing did not complete within {timeout:.1f} seconds "
        f"(joint_error={math.degrees(max_error):.2f} deg)"
    )


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
    parser.add_argument("--max-gripper-step-mm", type=float, default=cfg["max-gripper-step-mm"])
    parser.add_argument("--max-joint-lead-deg", type=float, default=cfg["max-joint-lead-deg"])
    parser.add_argument("--max-gripper-lead-mm", type=float, default=cfg["max-gripper-lead-mm"])
    parser.add_argument(
        "--max-act-raw-joint-gap-deg", type=float,
        default=cfg["max-act-raw-joint-gap-deg"],
        help="ACT raw target-to-feedback joint gap watchdog",
    )
    parser.add_argument(
        "--max-act-raw-gripper-gap-mm", type=float,
        default=cfg["max-act-raw-gripper-gap-mm"],
        help="ACT raw target-to-feedback gripper gap watchdog",
    )
    parser.add_argument(
        "--max-act-raw-gripper-bound-tolerance-mm", type=float,
        default=cfg["max-act-raw-gripper-bound-tolerance-mm"],
        help="ACT raw gripper target tolerance beyond [0, gripper-max-mm]",
    )
    parser.add_argument(
        "--max-act-clamp-hold-steps", type=int,
        default=cfg["max-act-clamp-hold-steps"],
        help="maximum ticks ACT may hold one rate-limited action",
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
        help="MOVE_J speed percentage for keyboard-Y homing",
    )
    parser.add_argument(
        "--home-timeout", type=float, default=cfg["home-timeout"],
        help="keyboard-Y homing timeout in seconds",
    )
    base.add_connection_arguments(parser, cfg)
    return parser.parse_args(argv)


def validate_motion_limits(args, policy_type=None):
    """Validate base limits and any policy-specific limits after overrides."""
    if args.max_joint_step_deg <= 0 or args.max_gripper_step_mm <= 0:
        raise RuntimeError("step limits must be positive")
    if args.max_joint_lead_deg < args.max_joint_step_deg:
        raise RuntimeError("max-joint-lead-deg must be at least max-joint-step-deg")
    if args.max_gripper_lead_mm < args.max_gripper_step_mm:
        raise RuntimeError("max-gripper-lead-mm must be at least max-gripper-step-mm")
    if args.max_act_clamp_hold_steps < 1:
        raise RuntimeError("max-act-clamp-hold-steps must be positive")
    for name in ("max_act_raw_joint_gap_deg", "max_act_raw_gripper_gap_mm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise RuntimeError(name.replace("_", "-") + " must be positive")
    if (
        args.max_act_raw_gripper_bound_tolerance_mm < 0
        or not math.isfinite(args.max_act_raw_gripper_bound_tolerance_mm)
    ):
        raise RuntimeError(
            "max-act-raw-gripper-bound-tolerance-mm must be finite "
            "and non-negative"
        )
    if policy_type in {"act", "pi0"} and (
        args.max_act_raw_joint_gap_deg is None
        or args.max_act_raw_gripper_gap_mm is None
    ):
        raise RuntimeError("ACT raw-action gap limits must be configured")


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
    args = parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    explicitly_set = {
        token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        for token in raw_argv
        if token.startswith("--")
    }
    for alias, destination in {"max_plan_age_ms": "max_smolvla_plan_age_ms",
                               "replan_lead_steps": "smolvla_replan_lead_steps"}.items():
        if alias in explicitly_set:
            explicitly_set.add(destination)
    if not 1 <= args.rate <= 20 or args.duration <= 0:
        raise RuntimeError("rate must be 1..20 Hz and duration must be positive")
    if not 1 <= args.camera_fps <= 60:
        raise RuntimeError("camera-fps must be 1..60")
    if not 1 <= args.warmup_steps <= 20:
        raise RuntimeError("warmup-steps must be 1..20")
    if not 0.2 <= args.keyboard_hold_seconds <= 5.0:
        raise RuntimeError("keyboard-hold-seconds must be in [0.2, 5.0]")
    if args.home_timeout <= 0:
        raise RuntimeError("home-timeout must be positive")
    if args.max_inference_ms <= 0 or args.max_camera_age_ms <= 0:
        raise RuntimeError("inference and camera-age limits must be positive")
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
            raise RuntimeError("Pi0 execution requires --experimental-pi0-execute; run shadow first")
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
            "max_gripper_step_mm": float(args.max_gripper_step_mm),
            "max_joint_lead_deg": float(args.max_joint_lead_deg),
            "max_gripper_lead_mm": float(args.max_gripper_lead_mm),
            **({
                "max_act_raw_joint_gap_deg": float(
                    args.max_act_raw_joint_gap_deg
                ),
                "max_act_raw_gripper_gap_mm": float(
                    args.max_act_raw_gripper_gap_mm
                ),
                "max_act_raw_gripper_bound_tolerance_mm": float(
                    args.max_act_raw_gripper_bound_tolerance_mm
                ),
                "max_act_clamp_hold_steps": args.max_act_clamp_hold_steps,
            } if policy_type in {"act", "pi0"} else {}),
            "gripper_max_mm": float(args.gripper_max_mm),
            "gripper_force": float(args.gripper_force),
            "max_camera_age_ms": float(args.max_camera_age_ms),
        }
        approval_checked = validate_execution_approval(
            args.approval.resolve(), digest, policy_type, approval_runtime
        )
        if not approval_checked:
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

    robot = None
    cameras = None
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    keyboard = keyboard_api.TerminalKeyEventReader()
    key_tracker = KeyboardIntentTracker(args.keyboard_hold_seconds)
    command_gate = False
    motion_enabled = False
    arm_prepared = False
    command_target = None
    held_act_action = None
    act_hold_steps = 0
    planner = None
    start_pending = False
    pending_plan_request_id = 0
    homing_active = False
    emergency_stopped = False
    # Piper JS is encoded as move_mode=J plus mit_mode=MIT. The controller's
    # mode_feedback therefore remains MOVE_J (1); MOVE_MIT (4/6) is reserved
    # for the separate per-joint MIT API.
    expected_js_mode = 1
    try:
        keyboard.start()
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
                policy.select_action(preprocessor(warm_batch))
        if callable(reset):
            reset()
        if policy_type != "pi0":
            print(f"Policy warm-up: {args.warmup_steps} steps complete", flush=True)
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
        if args.execute:
            print(
                "Keyboard: tap X to start/stop; hold "
                f"Y for {args.keyboard_hold_seconds:.1f}s to home and remain "
                "paused; Ctrl+C homes and exits; Alt+S immediately performs "
                "an electronic emergency stop. Other keys are disabled.",
                flush=True,
            )
            print("Camera window: X start/stop; hold Y home; S emergency stop; F fullscreen; Esc windowed.", flush=True)
        else:
            print(
                "Keyboard: X/Y/Alt+S motion controls are disabled in shadow "
                "mode; Ctrl+C exits without moving the arm.",
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
        with report.open("w", encoding="utf-8") as log:
            while args.execute or time.monotonic() < shadow_deadline:
                tick = time.monotonic()
                events = keyboard.poll()
                for key in cameras.poll_keys():
                    if key in (ord("x"), ord("X")):
                        events.append(("x", tick))
                    elif key in (ord("y"), ord("Y")):
                        events.append(("y", tick))
                    elif key in (ord("s"), ord("S")):
                        events.append(("alt+s", tick))
                x_toggle, home_requested, stop_requested = key_tracker.update(events, tick)
                trial_timed_out = (
                    args.execute
                    and motion_enabled
                    and trial_started_at is not None
                    and tick - trial_started_at >= args.duration
                )
                if trial_timed_out:
                    motion_enabled = False
                    start_pending = False
                    trial_started_at = None
                    active_actions.clear()
                    active_plan = None
                    planner_requested = False
                    command_target = None
                    held_act_action = None
                    act_hold_steps = 0
                    if planner is None and callable(reset):
                        reset()
                    print(
                        f"\nTrial {trial_id} reached {args.duration:g}s; "
                        "model control is paused. Hold Y to home or tap X "
                        "to start a new trial.",
                        flush=True,
                    )
                toggle_event = "trial_timeout" if trial_timed_out else None
                if stop_requested:
                    if not args.execute:
                        print("\nAlt+S ignored: electronic stop is disabled without --execute.")
                        continue
                    motion_enabled = False
                    emergency_stopped = True
                    robot.electronic_emergency_stop()
                    if planner is not None:
                        planner.close()
                        planner = None
                    print("\nAlt+S: electronic emergency stop sent; program exits.")
                    return 130
                if home_requested:
                    if not args.execute:
                        print(
                            "\nY ignored: homing is disabled without --execute."
                        )
                        continue
                    motion_enabled = False
                    start_pending = False
                    trial_started_at = None
                    active_actions.clear()
                    active_plan = None
                    planner_requested = False
                    held_act_action = None
                    act_hold_steps = 0
                    if planner is None and callable(reset):
                        reset()
                    print("\nY hold received: pausing model control and homing.")
                    homing_active = True
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
                    )
                    homing_active = False
                    arm_prepared = False
                    command_gate = False
                    command_target = None
                    print("Homing complete; model control remains paused. Tap X to resume.")
                    continue
                if args.execute and x_toggle:
                    motion_enabled = not motion_enabled
                    toggle_event = "start" if motion_enabled else "stop"
                    if motion_enabled:
                        trial_id += 1
                        trial_started_at = tick
                        print(f"\nTrial {trial_id} started.", flush=True)
                    else:
                        trial_started_at = None
                        active_actions.clear()
                        active_plan = None
                        planner_requested = False
                        command_target = None
                        print(f"\nTrial {trial_id} paused.", flush=True)
                    held_act_action = None
                    act_hold_steps = 0
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
                planner_underflow = False
                plan_observation_age_ms = None
                chunk_inference_ms = None
                act_action_held = False
                act_hold_timeout = False
                act_raw_action_ok = True
                act_raw_action_failures = []
                if planner is not None:
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
                    elif policy_type == "act" and held_act_action is not None:
                        # Do not consume the next ACT action while the robot is
                        # still catching up to the current rate-limited target.
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
                tracking_raw = (
                    act_tracking_action(
                        raw, args.gripper_max_mm / 1000.0,
                        args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                    ) if policy_type in {"act", "pi0"} else raw
                )
                target = clamp_step(
                    tracking_raw, state, math.radians(args.max_joint_step_deg),
                    args.max_gripper_step_mm / 1000.0, args.gripper_max_mm / 1000.0,
                    joint_bounds, command_target,
                    math.radians(args.max_joint_lead_deg),
                    args.max_gripper_lead_mm / 1000.0,
                )
                action_clamped = action_was_clamped(raw, target)
                tracking_clamped = action_was_clamped(tracking_raw, target)
                if policy_type in {"act", "pi0"}:
                    act_raw_action_failures = act_action_envelope_failures(
                        raw, state, joint_bounds,
                        args.gripper_max_mm / 1000.0,
                        math.radians(args.max_act_raw_joint_gap_deg),
                        args.max_act_raw_gripper_gap_mm / 1000.0,
                        args.max_act_raw_gripper_bound_tolerance_mm / 1000.0,
                    )
                    act_raw_action_ok = not act_raw_action_failures
                    if policy_type == "act" and args.execute and motion_enabled:
                        if tracking_clamped and act_raw_action_ok:
                            act_hold_steps = (
                                act_hold_steps + 1 if act_action_held else 1
                            )
                            held_act_action = np.asarray(raw).copy()
                        else:
                            held_act_action = None
                            act_hold_steps = 0
                        act_hold_timeout = (
                            act_hold_steps > args.max_act_clamp_hold_steps
                        )
                    else:
                        held_act_action = None
                        act_hold_steps = 0
                execution_ready = transition_is_safe(
                    joints, target[:6], joint_bounds
                )
                mode_feedback = read_mode_feedback(robot)
                mode_ready = not arm_prepared or mode_feedback == expected_js_mode
                healthy = (
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
                watchdog_failures = []
                if camera_age_ms > args.max_camera_age_ms:
                    watchdog_failures.append(
                        f"camera age {camera_age_ms:.1f}ms > "
                        f"{args.max_camera_age_ms:.1f}ms"
                    )
                if inference_ms > inference_limit_ms:
                    watchdog_failures.append(
                        f"inference {inference_ms:.1f}ms > "
                        f"{inference_limit_ms:.1f}ms"
                    )
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
                            f"ACT target remained clamped for {act_hold_steps} ticks > "
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
                # A watchdog trip always disarms the toggle. Recovery requires
                # another deliberate X press; motion must never restart itself.
                if (
                    args.execute and motion_enabled and not healthy
                    and not start_pending
                ):
                    motion_enabled = False
                    trial_started_at = None
                    toggle_event = "watchdog_stop"
                    held_act_action = None
                    if planner is None and callable(reset):
                        reset()
                    detail = "; ".join(watchdog_failures) or "unknown safety condition"
                    print(
                        f"\nWATCHDOG: Trial {trial_id} paused before the next "
                        f"command: {detail}",
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
                    )
                    arm_prepared = True
                    mode_feedback = expected_js_mode
                    if policy_type == "pi0":
                        # Mode setup may block for hundreds of ms. Revalidate
                        # timestamp and feedback next tick, never send that old action.
                        may_command = False
                if may_command and not command_gate:
                    command_gate = True
                if may_command:
                    robot.move_js(target[:6].tolist())
                    gripper.move_gripper_m(float(target[6]), args.gripper_force)
                    command_target = target.copy()
                elif command_gate:
                    robot.move_js(joints)
                    gripper.move_gripper_m(gripper_width, args.gripper_force)
                    command_gate = False
                    command_target = state.copy()
                elif args.execute:
                    command_target = state.copy()

                cameras.show(frames, args.execute and motion_enabled)
                control_loop_ms = (time.monotonic() - tick) * 1000.0
                control_deadline_missed = control_loop_ms > period * 1000.0

                row = {
                    "time": time.time(), "checkpoint_sha256": digest,
                    "policy_type": policy_type,
                    **({"pi0_clock_aligned": True,
                        "base_checkpoint": str(pi0_base)} if policy_type == "pi0" else {}),
                    "mode": "execute" if args.execute else "shadow",
                    "camera_age_ms": camera_age_ms, "inference_ms": inference_ms,
                    "inference_limit_ms": inference_limit_ms,
                    "rate_hz": args.rate,
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
                    "max_gripper_step_mm": args.max_gripper_step_mm,
                    "max_joint_lead_deg": args.max_joint_lead_deg,
                    "max_gripper_lead_mm": args.max_gripper_lead_mm,
                    "max_act_raw_joint_gap_deg": args.max_act_raw_joint_gap_deg,
                    "max_act_raw_gripper_gap_mm": args.max_act_raw_gripper_gap_mm,
                    "max_act_raw_gripper_bound_tolerance_mm": (
                        args.max_act_raw_gripper_bound_tolerance_mm
                    ),
                    "max_act_clamp_hold_steps": args.max_act_clamp_hold_steps,
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
                    **({
                        "act_tracking_action": np.asarray(tracking_raw).tolist(),
                        "act_tracking_clamped": tracking_clamped,
                        "act_gripper_boundary_corrected": bool(raw[6] != tracking_raw[6]),
                    } if policy_type in {"act", "pi0"} else {}),
                    "act_action_held": act_action_held,
                    "act_hold_steps": act_hold_steps,
                    "act_hold_timeout": act_hold_timeout,
                    "act_raw_action_ok": act_raw_action_ok,
                    "watchdog_failures": watchdog_failures,
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
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if not healthy:
                    mode = "WATCHDOG"
                elif may_command:
                    mode = "EXEC"
                else:
                    mode = "READY" if args.execute else "SHADOW"
                chunk_label = " refresh" if chunk_refresh else ""
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
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        print(f"\nReport: {report}")
        return 0
    except KeyboardInterrupt:
        if robot is not None and args.execute:
            motion_enabled = False
            print("\nCtrl+C received: pausing model control, homing, then exiting.")
            homing_active = True
            try:
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
                )
                homing_active = False
                arm_prepared = False
                print("Ctrl+C: homing complete; program exits.")
                return 0
            except KeyboardInterrupt:
                emergency_stopped = True
                robot.electronic_emergency_stop()
                print("\nSecond Ctrl+C during homing: electronic emergency stop sent.")
                return 130
            except Exception as exc:
                emergency_stopped = True
                robot.electronic_emergency_stop()
                print(
                    "\nCtrl+C homing failed; electronic emergency stop sent: "
                    f"{type(exc).__name__}: {exc}"
                )
                return 1
        print("\nCtrl+C: shadow inference exits without commanding the arm.")
        return 0
    except Exception:
        if robot is not None and homing_active:
            emergency_stopped = True
            robot.electronic_emergency_stop()
            print("\nHoming failed: electronic emergency stop sent.")
        raise
    finally:
        keyboard.close()
        if planner is not None:
            planner.close()
        if robot is not None and arm_prepared and not emergency_stopped:
            try:
                hold = read_valid_joints(robot, robot_config, 0.5)
                robot.move_js(hold)
            except Exception:
                pass
        if cameras is not None:
            cameras.close()
        base.disconnect_quietly(robot)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nOnline policy stopped: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
