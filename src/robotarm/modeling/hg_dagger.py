"""Human-gated DAgger support for the online Piper policy runner.

During autonomous execution the controlled leader shadows follower feedback.
On takeover the leader switches to zero-force mode; a clutch offset makes the
first expert request equal the follower's measured pose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..core.robot import piper as base
from ..teleop.leader_follower.runtime import (
    command_joint_bounds,
    ensure_receive_running,
    exit_leader_drag_mode,
    home_leader_and_restore,
    map_leader_joints,
    prepare_leader_target,
    read_gripper_message,
    read_joint_message,
    read_leader_gripper_command,
    update_leader_joint_message,
    wait_initial_leader_joints,
    wait_leader_gripper_command,
)


@dataclass
class ExpertSample:
    requested_action: list[float]
    leader_state: list[float]
    feedback_age: float


class LeaderHomingFailure(RuntimeError):
    """The physical leader could not complete its verified firmware home."""


class LeaderHomingNoMotion(LeaderHomingFailure):
    """The controlled leader accepted a home command but did not move."""


class TakeoverMotionGate:
    """Latch once deliberate leader motion is observed after clutching."""

    def __init__(
        self,
        joint_threshold: float,
        gripper_threshold: float,
        consecutive_samples: int,
    ) -> None:
        if not math.isfinite(joint_threshold) or joint_threshold <= 0:
            raise ValueError("joint motion threshold must be positive")
        if not math.isfinite(gripper_threshold) or gripper_threshold <= 0:
            raise ValueError("gripper motion threshold must be positive")
        if consecutive_samples < 1:
            raise ValueError("consecutive motion samples must be positive")
        self.joint_threshold = float(joint_threshold)
        self.gripper_threshold = float(gripper_threshold)
        self.consecutive_samples = int(consecutive_samples)
        self.reference: list[float] | None = None
        self.streak = 0
        self.triggered = False

    def arm(self, requested_action: list[float]) -> None:
        if len(requested_action) != 7 or not all(
            math.isfinite(value) for value in requested_action
        ):
            raise ValueError("takeover reference action must be finite and 7D")
        self.reference = [float(value) for value in requested_action]
        self.streak = 0
        self.triggered = False

    def update(self, requested_action: list[float]) -> bool:
        if self.reference is None:
            raise RuntimeError("takeover motion gate is not armed")
        if len(requested_action) != 7 or not all(
            math.isfinite(value) for value in requested_action
        ):
            raise ValueError("expert requested action must be finite and 7D")
        if self.triggered:
            return True
        joint_motion = max(
            abs(float(value) - reference)
            for value, reference in zip(requested_action[:6], self.reference[:6])
        )
        gripper_motion = abs(
            float(requested_action[6]) - self.reference[6]
        )
        moved = (
            joint_motion >= self.joint_threshold
            or gripper_motion >= self.gripper_threshold
        )
        self.streak = self.streak + 1 if moved else 0
        self.triggered = self.streak >= self.consecutive_samples
        return self.triggered

    def reset(self) -> None:
        self.reference = None
        self.streak = 0
        self.triggered = False


class ClutchedLeaderMapping:
    """Map absolute leader feedback to continuous follower-relative targets."""

    def __init__(self, signs: list[int], offsets_deg: list[float], gripper_max: float):
        if len(signs) != 6 or any(value not in (-1, 1) for value in signs):
            raise ValueError("HG-DAgger joint signs must contain six -1/+1 values")
        if len(offsets_deg) != 6 or not all(math.isfinite(v) for v in offsets_deg):
            raise ValueError("HG-DAgger joint offsets must contain six finite values")
        if not math.isfinite(gripper_max) or gripper_max <= 0:
            raise ValueError("HG-DAgger gripper maximum must be positive")
        self.signs = list(signs)
        self.offsets_deg = list(offsets_deg)
        self.gripper_max = float(gripper_max)
        self._joint_clutch: list[float] | None = None
        self._gripper_clutch: float | None = None

    def engage(self, leader_state: list[float], follower_state: list[float]) -> None:
        if len(leader_state) != 7 or len(follower_state) != 7:
            raise ValueError("leader and follower states must both be 7D")
        mapped = map_leader_joints(
            leader_state[:6], self.signs, self.offsets_deg
        )
        self._joint_clutch = [
            follower - leader for follower, leader in zip(follower_state[:6], mapped)
        ]
        leader_width = min(max(float(leader_state[6]), 0.0), self.gripper_max)
        follower_width = min(max(float(follower_state[6]), 0.0), self.gripper_max)
        self._gripper_clutch = follower_width - leader_width

    def target(
        self,
        leader_state: list[float],
        bounds: list[tuple[float, float]],
    ) -> list[float]:
        if self._joint_clutch is None or self._gripper_clutch is None:
            raise RuntimeError("HG-DAgger clutch is not engaged")
        mapped = map_leader_joints(
            leader_state[:6], self.signs, self.offsets_deg
        )
        requested = [
            value + offset for value, offset in zip(mapped, self._joint_clutch)
        ]
        requested = prepare_leader_target(requested, bounds, control_only=False)
        leader_width = min(max(float(leader_state[6]), 0.0), self.gripper_max)
        gripper = min(
            max(leader_width + self._gripper_clutch, 0.0),
            self.gripper_max,
        )
        # Rebase at a saturated endpoint. Otherwise a 79 mm clutch offset
        # survives opening fully, and closing the leader can never request
        # less than 79 mm on the follower. Repeated identical samples remain
        # stable, and engagement still requests the measured follower width.
        self._gripper_clutch = gripper - leader_width
        return [*requested, gripper]


class FollowingLeaderMapping:
    """Make a controlled leader shadow follower motion without an engage jump."""

    def __init__(self, signs: list[int], offsets_deg: list[float], gripper_max: float):
        if len(signs) != 6 or any(value not in (-1, 1) for value in signs):
            raise ValueError("HG-DAgger joint signs must contain six -1/+1 values")
        if len(offsets_deg) != 6 or not all(math.isfinite(v) for v in offsets_deg):
            raise ValueError("HG-DAgger joint offsets must contain six finite values")
        if not math.isfinite(gripper_max) or gripper_max <= 0:
            raise ValueError("HG-DAgger gripper maximum must be positive")
        self.signs = list(signs)
        self.offsets_deg = list(offsets_deg)
        self.gripper_max = float(gripper_max)
        self._joint_offset: list[float] | None = None
        self._gripper_offset: float | None = None

    def _inverse_joints(self, follower_joints: list[float]) -> list[float]:
        if len(follower_joints) != 6:
            raise ValueError("follower joints must contain six values")
        return [
            sign * (value - math.radians(offset))
            for value, sign, offset in zip(
                follower_joints, self.signs, self.offsets_deg
            )
        ]

    def engage(self, leader_state: list[float], follower_state: list[float]) -> None:
        if len(leader_state) != 7 or len(follower_state) != 7:
            raise ValueError("leader and follower states must both be 7D")
        inverse = self._inverse_joints(follower_state[:6])
        self._joint_offset = [
            leader - target for leader, target in zip(leader_state[:6], inverse)
        ]
        leader_width = min(max(float(leader_state[6]), 0.0), self.gripper_max)
        follower_width = min(max(float(follower_state[6]), 0.0), self.gripper_max)
        self._gripper_offset = leader_width - follower_width

    def target(
        self,
        follower_state: list[float],
        leader_bounds: list[tuple[float, float]],
    ) -> list[float]:
        if self._joint_offset is None or self._gripper_offset is None:
            raise RuntimeError("leader following reference is not engaged")
        if len(follower_state) != 7 or not all(
            math.isfinite(value) for value in follower_state
        ):
            raise ValueError("follower state must be finite and 7D")
        inverse = self._inverse_joints(follower_state[:6])
        joints = [
            value + offset for value, offset in zip(inverse, self._joint_offset)
        ]
        joints = prepare_leader_target(joints, leader_bounds, control_only=False)
        follower_width = min(max(float(follower_state[6]), 0.0), self.gripper_max)
        gripper = min(
            max(follower_width + self._gripper_offset, 0.0),
            self.gripper_max,
        )
        self._gripper_offset = gripper - follower_width
        return [*joints, gripper]

    def absolute_target(
        self,
        follower_state: list[float],
        leader_bounds: list[tuple[float, float]],
    ) -> list[float]:
        """Map follower feedback to the matching absolute leader pose."""
        if len(follower_state) != 7 or not all(
            math.isfinite(value) for value in follower_state
        ):
            raise ValueError("follower state must be finite and 7D")
        joints = prepare_leader_target(
            self._inverse_joints(follower_state[:6]),
            leader_bounds,
            control_only=False,
        )
        gripper = min(max(float(follower_state[6]), 0.0), self.gripper_max)
        return [*joints, gripper]

    def reset(self) -> None:
        self._joint_offset = None
        self._gripper_offset = None

    @property
    def engaged(self) -> bool:
        return self._joint_offset is not None and self._gripper_offset is not None


class LeaderExpert:
    """Own the leader connection and expose clutched expert targets."""

    def __init__(
        self,
        *,
        interface: str,
        channel: str,
        firmware: str,
        feedback_timeout: float,
        init_timeout: float,
        signs: list[int],
        offsets_deg: list[float],
        gripper_max: float,
        gripper_force: float,
        initial_gripper_width: float | None = None,
    ) -> None:
        self.feedback_timeout = float(feedback_timeout)
        self.channel = str(channel)
        self.gripper_force = float(gripper_force)
        self.robot = None
        self.gripper = None
        self.config = None
        self.drag_active = False
        self.controlled_ready = False
        self.homing_active = False
        self._homing_target: list[float] | None = None
        self.homing_error_degrees: float | None = None
        self._homing_started_at: float | None = None
        self._homing_initial_error: float | None = None
        self.joints: list[float] | None = None
        self.gripper_width = 0.0
        self.gripper_not_before = 0.0
        self.mapping = ClutchedLeaderMapping(signs, offsets_deg, gripper_max)
        self.follow_mapping = FollowingLeaderMapping(
            signs, offsets_deg, gripper_max
        )
        if initial_gripper_width is not None and (
            not math.isfinite(initial_gripper_width)
            or not 0.0 <= initial_gripper_width <= gripper_max
        ):
            raise ValueError(
                "initial leader gripper width must be within the configured range"
            )
        args = SimpleNamespace(
            interface=interface,
            channel=channel,
            firmware=firmware,
            feedback_timeout=feedback_timeout,
        )
        try:
            self.robot, self.config = base.connect_robot(args)
            status_deadline = time.monotonic() + min(
                init_timeout, max(2.0, self.feedback_timeout * 10.0)
            )
            status = None
            while time.monotonic() < status_deadline:
                status = self.robot.get_arm_status()
                if status is not None:
                    break
                time.sleep(0.02)
            if status is None:
                raise RuntimeError(
                    "HG-DAgger leader arm status did not arrive after CAN connect"
                )
            arm_status = base.value_as_int(status.msg.arm_status)
            if arm_status == 1:
                print(
                    "[HG-DAgger] leader is latched in electronic-stop state; "
                    "resetting motion state before controlled startup.",
                    flush=True,
                )
                self.robot.reset()
                deadline = time.monotonic() + max(2.0, self.feedback_timeout * 10.0)
                while time.monotonic() < deadline:
                    status = self.robot.get_arm_status()
                    if (
                        status is not None
                        and base.value_as_int(status.msg.arm_status) == 0
                    ):
                        break
                    time.sleep(0.02)
                else:
                    raise RuntimeError(
                        "HG-DAgger leader electronic-stop state did not reset"
                    )
            elif arm_status != 0:
                raise RuntimeError(
                    "HG-DAgger leader status is abnormal "
                    f"(arm_status={arm_status}); automatic reset refused"
                )
            # Match the follower deployment connection. Project-level bounds
            # remain enforced; disabling the SDK's canonical limits avoids
            # clamping this arm's measured J2/J3 zero offsets.
            self.robot.set_joint_limits_enabled(False)
            self.robot.set_auto_set_motion_mode_enabled(False)
            self.joints, source = wait_initial_leader_joints(
                self.robot, self.config, init_timeout
            )
            self.drag_active = source == "drag_input"
            self.controlled_ready = source == "feedback"
            print(
                "[HG-DAgger] leader initial source: "
                + (
                    "normal feedback; startup will explicitly arm target-control mode."
                    if self.controlled_ready
                    else "zero-force drag input; controlled mode must be restored."
                ),
                flush=True,
            )
            self.gripper = self.robot.init_effector(
                self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            gripper_feedback_available = True
            try:
                self.gripper_width, _ = read_gripper_message(
                    self.gripper, "HG-DAgger leader"
                )
            except RuntimeError:
                gripper_feedback_available = False
            if source == "drag_input":
                self.drag_active = True
            has_safe_gripper_fallback = initial_gripper_width is not None
            if not gripper_feedback_available and has_safe_gripper_fallback:
                self.gripper_width = float(initial_gripper_width)
                print(
                    "[HG-DAgger] leader gripper feedback is unavailable; using "
                    "the follower's measured width as the initial hold target "
                    f"({self.gripper_width * 1000.0:.1f} mm).",
                    flush=True,
                )
            elif not gripper_feedback_available and not self.drag_active:
                print(
                    "[HG-DAgger] leader gripper feedback is unavailable; entering "
                    "zero-force mode to acquire a fresh gripper input. Support "
                    "the leader arm.",
                    flush=True,
                )
                self.robot.set_leader_mode()
                time.sleep(0.1)
                self.drag_active = True
            if self.drag_active:
                self.gripper_not_before = time.time()
            if not gripper_feedback_available and not has_safe_gripper_fallback:
                print(
                    "[HG-DAgger] waiting for a new leader gripper input; "
                    "open or close the leader gripper once.",
                    flush=True,
                )
                try:
                    self.gripper_width, _ = wait_leader_gripper_command(
                        self.gripper,
                        init_timeout,
                        not_before=self.gripper_not_before,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    # Restore the arm role on failure, but do not command an
                    # unknown gripper width during cleanup.
                    self.gripper = None
                    raise RuntimeError(
                        "HG-DAgger received no valid leader gripper input; "
                        f"open or close it while the prompt is shown "
                        f"(timeout {init_timeout:.1f}s)"
                    ) from exc
            if self.drag_active:
                self.joints, _ = update_leader_joint_message(
                    self.robot, self.joints, self.config
                )
            ensure_receive_running(self.robot, "HG-DAgger leader")
        except BaseException:
            self.close()
            raise

    def _read(self) -> tuple[list[float], float]:
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None and self.joints is not None
        ensure_receive_running(self.robot, "HG-DAgger leader")
        self.joints, joint_age = update_leader_joint_message(
            self.robot, self.joints, self.config
        )
        self.gripper_width, gripper_age = read_leader_gripper_command(
            self.gripper,
            self.gripper_width,
            not_before=self.gripper_not_before,
        )
        finite_ages = [age for age in (joint_age, gripper_age) if math.isfinite(age)]
        age = max(finite_ages) if finite_ages else -1.0
        return [*self.joints, self.gripper_width], age

    def _read_controlled(self) -> tuple[list[float], float]:
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None
        self.joints, joint_age = read_joint_message(
            self.robot, self.config, "HG-DAgger leader"
        )
        self.gripper_width, gripper_age = read_gripper_message(
            self.gripper, "HG-DAgger leader"
        )
        age = max(joint_age, gripper_age)
        if not math.isfinite(age) or age > self.feedback_timeout:
            raise RuntimeError(
                f"HG-DAgger controlled leader feedback is stale ({age * 1000:.0f} ms)"
            )
        return [*self.joints, self.gripper_width], age

    def _wait_controlled(self, timeout: float | None = None) -> tuple[list[float], float]:
        """Wait for post-role-switch joint and gripper feedback, not cached frames."""
        timeout = (
            max(2.0, self.feedback_timeout * 10.0)
            if timeout is None
            else float(timeout)
        )
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._read_controlled()
            except RuntimeError as exc:
                last_error = exc
            time.sleep(0.01)
        detail = "feedback unavailable" if last_error is None else str(last_error)
        raise TimeoutError(
            "HG-DAgger leader did not publish fresh controlled feedback within "
            f"{timeout:.1f}s after leaving zero-force mode: {detail}"
        )

    def hold_controlled(self) -> None:
        """Arm external target control at the measured pose and hold there."""
        if self.robot is None or self.joints is None:
            return
        if self.drag_active:
            leader_state, _ = self._read()
        else:
            leader_state = [*self.joints, self.gripper_width]
        # Piper calls the external-target state "follower mode".  Reassert it
        # even when ordinary 0x2A feedback is already present: feedback alone
        # does not prove that the controller will consume MOVE_J/JS targets.
        # exit_leader_drag_mode commands only the measured pose while changing
        # state, so this does not create a leader/follower linkage or a jump.
        exit_leader_drag_mode(
            self.robot,
            leader_state[:6],
            self.gripper,
            leader_state[6],
            self.gripper_force,
            max(2.0, self.feedback_timeout * 10.0),
            self.config,
        )
        self.drag_active = False
        self.controlled_ready = True
        self.follow_mapping.reset()

    def resume_following(self, follower_state: list[float]) -> None:
        """Enter controlled mode and latch a no-jump leader/follower offset."""
        self.hold_controlled()
        leader_state, _ = self._wait_controlled()
        self.follow_mapping.engage(leader_state, follower_state)
        self.follow(follower_state)

    def resume_policy_control(self, follower_state: list[float]) -> None:
        """Resume policy motion from each arm's measured pose without realignment."""
        print("[HG-DAgger] restoring leader role, joint enable and gripper enable...", flush=True)
        self.hold_controlled()
        print("[HG-DAgger] leader enabled; waiting for controlled feedback...", flush=True)
        leader_state, _ = self._wait_controlled()
        assert self.robot is not None
        # Role changes and enable broadcasts can reset controller settings.
        # Match the follower's policy stream setup AFTER enabling, while the
        # only loaded target is the leader's own measured pose.
        self.robot.set_speed_percent(100)
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.JS)
        self.robot.move_js(leader_state[:6])
        self._wait_policy_control()
        print("[HG-DAgger] leader policy stream ready.", flush=True)
        self.follow_mapping.engage(leader_state, follower_state)
        self.command_policy_action(follower_state)

    def _wait_policy_control(self) -> None:
        """Require controller acknowledgement before returning control to ACT."""
        deadline = time.monotonic() + max(2.0, self.feedback_timeout * 10.0)
        detail = "no status feedback"
        while time.monotonic() < deadline:
            status = self.robot.get_arm_status()
            enabled = self.robot.get_joints_enable_status_list()
            if status is not None:
                ctrl = base.value_as_int(status.msg.ctrl_mode)
                mode = base.value_as_int(status.msg.mode_feedback)
                arm = base.value_as_int(status.msg.arm_status)
                detail = f"ctrl_mode={ctrl}, mode_feedback={mode}, arm_status={arm}, enabled={enabled}"
                # Piper JS uses CAN control (1), joint mode feedback (1).
                if ctrl == 1 and mode == 1 and arm == 0 and len(enabled) == 6 and all(enabled):
                    return
            time.sleep(0.02)
        raise RuntimeError("HG-DAgger leader policy control was not restored: " + detail)

    def command_policy_action(self, follower_action: list[float]) -> ExpertSample:
        """Send one follower-coordinate ACT target to the controlled leader."""
        if getattr(self, "homing_active", False):
            raise RuntimeError("cannot command policy action while leader homing owns control")
        if self.drag_active:
            raise RuntimeError("cannot command policy action while leader is zero-force")
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None
        leader_state, age = self._read_controlled()
        mapping = (
            self.follow_mapping.target if self.follow_mapping.engaged
            else self.follow_mapping.absolute_target
        )
        target = mapping(
            follower_action, command_joint_bounds(self.config, 1.0)
        )
        self.robot.move_js(target[:6])
        self.gripper.move_gripper_m(target[6], self.gripper_force)
        return ExpertSample(target, leader_state, age)

    def align_to_follower(
        self,
        follower_state: list[float],
        *,
        speed_percent: float,
        timeout: float,
        stop_requested: Any = None,
    ) -> None:
        """Move the controlled leader to the follower's matching absolute pose."""
        if not math.isfinite(speed_percent) or not 1.0 <= speed_percent <= 30.0:
            raise ValueError("leader alignment speed must be within 1..30 percent")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("leader alignment timeout must be positive")
        self.hold_controlled()
        leader_state, _ = self._wait_controlled()
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None
        target = self.follow_mapping.absolute_target(
            follower_state, command_joint_bounds(self.config, 1.0)
        )
        initial_error = max(
            abs(actual - requested)
            for actual, requested in zip(leader_state[:6], target[:6])
        )
        print(
            "[HG-DAgger] aligning controlled leader to follower: "
            f"max joint error={math.degrees(initial_error):.1f} deg at "
            f"{speed_percent:g}% speed.",
            flush=True,
        )
        self.robot.set_speed_percent(speed_percent)
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.J)
        self.robot.move_j(leader_state[:6])
        self.gripper.move_gripper_m(target[6], self.gripper_force)
        # Automatic mode frames are disabled for streaming safety.  The
        # current-pose hold above consumes the planned-J trigger, so re-arm it
        # immediately before the actual home target.
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.J)
        self.robot.move_j(target[:6])
        deadline = time.monotonic() + timeout
        settled_at = None
        while time.monotonic() < deadline:
            if stop_requested is not None and stop_requested():
                self.robot.electronic_emergency_stop()
                raise InterruptedError(
                    "electronic emergency stop requested during leader alignment"
                )
            measured, _ = self._wait_controlled(
                min(1.0, max(self.feedback_timeout * 5.0, 0.25))
            )
            joint_error = max(
                abs(actual - requested)
                for actual, requested in zip(measured[:6], target[:6])
            )
            gripper_error = abs(measured[6] - target[6])
            if joint_error <= math.radians(2.0) and gripper_error <= 0.005:
                if settled_at is None:
                    settled_at = time.monotonic()
                elif time.monotonic() - settled_at >= 0.3:
                    self.joints = measured[:6]
                    self.gripper_width = measured[6]
                    self.robot.move_j(self.joints)
                    self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.JS)
                    self.robot.move_js(self.joints)
                    self.follow_mapping.engage(measured, follower_state)
                    self.follow(follower_state)
                    print(
                        "[HG-DAgger] leader aligned with follower and holding.",
                        flush=True,
                    )
                    return
            else:
                settled_at = None
            time.sleep(0.05)
        raise TimeoutError(
            "HG-DAgger leader alignment did not reach the follower pose within "
            f"{timeout:.1f}s"
        )

    def _wait_home_joints(self, timeout: float) -> list[float]:
        """Require two fresh, stable, bounded snapshots before native home."""
        started = time.monotonic()
        deadline = started + timeout
        previous = None
        last_error = "no complete joint feedback"
        while time.monotonic() < deadline:
            try:
                joints, age = read_joint_message(
                    self.robot, self.config, "HG-DAgger leader before home"
                )
                elapsed = time.monotonic() - started
                if not math.isfinite(age) or age < 0 or age > self.feedback_timeout or age > elapsed:
                    last_error = "waiting for all joint pairs newer than the homing preflight"
                else:
                    if previous is not None and max(abs(a - b) for a, b in zip(joints, previous)) <= math.radians(2):
                        return list(joints)
                    previous = list(joints)
                    # The next accepted snapshot must also contain new frames.
                    started = time.monotonic()
                    last_error = "waiting for a second stable joint snapshot"
            except RuntimeError as exc:
                previous = None
                last_error = str(exc)
            time.sleep(0.02)
        raise LeaderHomingFailure(
            f"HG-DAgger leader home preflight did not receive fresh valid feedback "
            f"within {timeout:.1f}s; no leader homing/model target sent: {last_error}"
        )

    def start_homing(
        self,
        follower_target: list[float],
        speed_percent: float,
        *,
        timeout: float = 60.0,
        stop_requested: Any = None,
    ) -> None:
        """Home the physical leader with firmware 0x191, then restore control."""
        if not math.isfinite(speed_percent) or not 1.0 <= speed_percent <= 30.0:
            raise ValueError("leader homing speed must be within 1..30 percent")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("leader homing timeout must be positive")
        stop_requested = (lambda: False) if stop_requested is None else stop_requested
        if self.drag_active:
            leader_state, _ = self._read()
        else:
            # Homing needs the measured joints, not Follower-role gripper
            # feedback. Keep the initial gripper hold width from collection.
            joints = self._wait_home_joints(min(timeout, max(2.0, self.feedback_timeout)))
            leader_state = [*joints, self.gripper_width]
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None
        target = self.follow_mapping.absolute_target(
            follower_target, command_joint_bounds(self.config, 1.0)
        )
        self.follow_mapping.reset()
        self._homing_target = target
        self.homing_active = True
        self.homing_error_degrees = math.degrees(
            max(
                abs(actual - requested)
                for actual, requested in zip(leader_state[:6], target[:6])
            )
        )
        self._homing_started_at = time.monotonic()
        self._homing_initial_error = self.homing_error_degrees
        print(
            "[HG-DAgger] starting physical leader home with verified firmware "
            f"0x191: max error={self.homing_error_degrees:.1f} deg.",
            flush=True,
        )
        try:
            # Match collection: firmware 0x191 home requires the Leader role.
            # Controlled policy execution uses Follower, so restore Leader
            # before native home and return to target control only afterwards.
            if not self.drag_active and max(abs(v) for v in leader_state[:6]) > math.radians(3.0):
                self.robot.set_leader_mode()
                self.drag_active = True
                self.controlled_ready = False
                time.sleep(0.1)
            homed_joints = home_leader_and_restore(
                self.robot,
                self.config,
                self.channel,
                leader_state[:6],
                timeout,
                stop_requested,
                move_j_speed_percent=speed_percent,
            )
            exit_leader_drag_mode(
                self.robot,
                homed_joints,
                self.gripper,
                target[6],
                self.gripper_force,
                self.feedback_timeout,
                self.config,
            )
            self.drag_active = False
            self.controlled_ready = True
            measured, _ = self._wait_controlled()
            self.joints = measured[:6]
            self.gripper_width = measured[6]
            self.homing_error_degrees = math.degrees(
                max(
                    abs(actual - requested)
                    for actual, requested in zip(
                        measured[:6], self._homing_target[:6]
                    )
                )
            )
        except (KeyboardInterrupt, InterruptedError):
            raise
        except Exception as exc:
            raise LeaderHomingFailure(
                "HG-DAgger verified firmware leader home failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print(
            "[HG-DAgger] leader firmware home verified; waiting for the "
            "concurrent follower MOVE_J home.",
            flush=True,
        )

    def homing_ready(self) -> bool:
        """Return true after the leader reaches its planned home target."""
        if not self.homing_active or self._homing_target is None:
            return True
        measured, _ = self._read_controlled()
        self.joints = measured[:6]
        self.gripper_width = measured[6]
        error = max(
            abs(actual - requested)
            for actual, requested in zip(measured[:6], self._homing_target[:6])
        )
        self.homing_error_degrees = math.degrees(error)
        if (
            self._homing_started_at is not None
            and self._homing_initial_error is not None
            and self._homing_initial_error > 2.0
            and time.monotonic() - self._homing_started_at >= 8.0
            and self.homing_error_degrees
            > self._homing_initial_error - 0.5
        ):
            status = self.robot.get_arm_status()
            ctrl_mode = (
                "unavailable"
                if status is None
                else getattr(status.msg, "ctrl_mode", "unavailable")
            )
            arm_status = (
                "unavailable"
                if status is None
                else getattr(status.msg, "arm_status", "unavailable")
            )
            mode_feedback = (
                "unavailable"
                if status is None
                else getattr(status.msg, "mode_feedback", "unavailable")
            )
            enabled = self.robot.get_joints_enable_status_list()
            raise LeaderHomingNoMotion(
                "HG-DAgger leader homing command produced no motion within 8s "
                f"(error stuck at {self.homing_error_degrees:.2f} deg; "
                f"enabled={enabled}; ctrl_mode={ctrl_mode}; "
                f"arm_status={arm_status}; mode_feedback={mode_feedback})"
            )
        return error <= math.radians(3.0)

    def finish_homing(self, follower_state: list[float]) -> None:
        """Switch the homed leader to JS shadow following without a jump."""
        if not self.homing_active:
            return
        measured, _ = self._wait_controlled()
        assert self.robot is not None
        self.joints = measured[:6]
        self.gripper_width = measured[6]
        self.robot.move_j(self.joints)
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.JS)
        self.robot.move_js(self.joints)
        self._homing_target = None
        self.homing_active = False
        self.homing_error_degrees = None
        self._homing_started_at = None
        self._homing_initial_error = None
        self.follow_mapping.reset()
        self.command_policy_action(follower_state)
        print(
            "[HG-DAgger] both arms reached home; dual-policy control ready.",
            flush=True,
        )

    def emergency_stop(self) -> None:
        try:
            if self.robot is not None:
                self.robot.electronic_emergency_stop()
        finally:
            # Cleanup must never restore hold/enable after an explicit stop,
            # including when the stop send itself fails on a disconnected bus.
            self.drag_active = False
            self.controlled_ready = False
        self.homing_active = False
        self._homing_target = None
        self.homing_error_degrees = None
        self._homing_started_at = None
        self._homing_initial_error = None

    def follow(self, follower_state: list[float]) -> ExpertSample:
        """Command the controlled leader from follower feedback."""
        if self.drag_active:
            raise RuntimeError("cannot shadow follower while leader is in zero-force mode")
        assert self.robot is not None and self.config is not None
        assert self.gripper is not None
        leader_state, age = self._read_controlled()
        if not self.follow_mapping.engaged:
            self.follow_mapping.engage(leader_state, follower_state)
        target = self.follow_mapping.target(
            follower_state, command_joint_bounds(self.config, 1.0)
        )
        self.robot.move_js(target[:6])
        self.gripper.move_gripper_m(target[6], self.gripper_force)
        return ExpertSample(target, leader_state, age)

    def engage(
        self,
        follower_state: list[float],
        bounds: list[tuple[float, float]],
    ) -> ExpertSample:
        assert self.robot is not None
        if not self.drag_active:
            leader_state, _ = self._read_controlled()
            self.joints = list(leader_state[:6])
            self.gripper_width = float(leader_state[6])
            # Release the position servo BEFORE changing the arm role. Merely
            # stopping width commands leaves the previous enabled target active
            # across the firmware transition. The leader gripper must be passive
            # while the human operates it; exit_leader_drag_mode re-enables it.
            assert self.gripper is not None
            base.set_gripper_enabled(
                self.gripper, False, self.gripper_width,
                self.gripper_force, max(2.0, self.feedback_timeout),
            )
            # Discard cached input from the previous drag session. Do not
            # transmit a gripper motor command after handing it to the human.
            self.gripper_not_before = time.time()
            self.robot.set_leader_mode()
            time.sleep(0.1)
            self.drag_active = True
            self.controlled_ready = False
            print(
                "[HG-DAgger] leader entered zero-force mode; "
                "gripper handed to human control.",
                flush=True,
            )
        self.follow_mapping.reset()
        leader_state, age = self._read()
        self.mapping.engage(leader_state, follower_state)
        return ExpertSample(
            self.mapping.target(leader_state, bounds),
            leader_state,
            age,
        )

    def sample(
        self, bounds: list[tuple[float, float]],
    ) -> ExpertSample:
        leader_state, age = self._read()
        return ExpertSample(self.mapping.target(leader_state, bounds), leader_state, age)

    def close(self) -> None:
        robot = self.robot
        if robot is not None and self.drag_active and self.joints is not None:
            try:
                exit_leader_drag_mode(
                    robot,
                    self.joints,
                    self.gripper,
                    self.gripper_width,
                    self.gripper_force,
                    self.feedback_timeout,
                    self.config,
                )
                print("[HG-DAgger] leader left zero-force mode and is holding.")
            except Exception as exc:
                print(
                    "[HG-DAgger] WARNING: could not restore leader hold: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        self.drag_active = False
        self.controlled_ready = False
        self.homing_active = False
        self._homing_target = None
        self.homing_error_degrees = None
        self._homing_started_at = None
        self._homing_initial_error = None
        base.disconnect_quietly(robot)
        self.robot = None

