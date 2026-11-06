"""Human-gated DAgger support for the online Piper policy runner.

The leader arm stays in zero-force mode.  On takeover, a clutch offset makes
the first expert request equal the follower's measured pose, so switching
control cannot create an absolute-pose jump.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..core.robot import piper as base
from ..teleop.leader_follower.runtime import (
    ensure_receive_running,
    exit_leader_drag_mode,
    map_leader_joints,
    prepare_leader_target,
    read_gripper_message,
    read_leader_gripper_command,
    update_leader_joint_message,
    wait_initial_leader_joints,
)


@dataclass
class ExpertSample:
    requested_action: list[float]
    leader_state: list[float]
    feedback_age: float


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
        self._gripper_clutch = float(follower_state[6] - leader_state[6])

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
        gripper = min(
            max(float(leader_state[6] + self._gripper_clutch), 0.0),
            self.gripper_max,
        )
        return [*requested, gripper]


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
    ) -> None:
        self.feedback_timeout = float(feedback_timeout)
        self.gripper_force = float(gripper_force)
        self.robot = None
        self.gripper = None
        self.config = None
        self.drag_active = False
        self.joints: list[float] | None = None
        self.gripper_width = 0.0
        self.gripper_not_before = 0.0
        self.mapping = ClutchedLeaderMapping(signs, offsets_deg, gripper_max)
        args = SimpleNamespace(
            interface=interface,
            channel=channel,
            firmware=firmware,
            feedback_timeout=feedback_timeout,
        )
        try:
            self.robot, self.config = base.connect_robot(args)
            self.joints, source = wait_initial_leader_joints(
                self.robot, self.config, init_timeout
            )
            self.drag_active = source == "drag_input"
            self.gripper = self.robot.init_effector(
                self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            try:
                self.gripper_width, _ = read_gripper_message(
                    self.gripper, "HG-DAgger leader"
                )
            except RuntimeError:
                value, age = read_leader_gripper_command(
                    self.gripper, self.gripper_width
                )
                if not math.isfinite(age):
                    # Restore the arm role on failure, but do not command an
                    # unknown gripper width during cleanup.
                    self.gripper = None
                    raise RuntimeError(
                        "HG-DAgger leader gripper feedback is unavailable; "
                        "operate the leader gripper once and retry"
                    )
                self.gripper_width = value
            if source != "drag_input":
                print(
                    "[HG-DAgger] entering leader zero-force mode; support the leader arm.",
                    flush=True,
                )
                self.robot.set_leader_mode()
                time.sleep(0.1)
            self.drag_active = True
            self.gripper_not_before = time.time()
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

    def engage(
        self,
        follower_state: list[float],
        bounds: list[tuple[float, float]],
    ) -> ExpertSample:
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
        base.disconnect_quietly(robot)
        self.robot = None

