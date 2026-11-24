#!/usr/bin/env python3
"""Reset the software emergency-stop state on two Piper arms in one command.

This sends the SDK motion-controller reset command to both selected CAN
controllers.  It does not release a physical emergency-stop button.  Reset may
remove holding torque, so execution requires ``--confirm-drop --execute``.
"""

from __future__ import annotations

import argparse
import math
import time
from types import SimpleNamespace
from typing import Any

from ..config import defaults
from ..core.robot import piper as base


def make_parser() -> argparse.ArgumentParser:
    cfg = defaults("data_acquisition", "enable-control")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--interface", default=cfg.get("interface") or "socketcan")
    parser.add_argument("--follower-channel", default="can_follower1")
    parser.add_argument("--leader-channel", default="can_leader1")
    parser.add_argument(
        "--firmware",
        choices=("default", "v183", "v188", "v189"),
        default="v189",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=base.positive_float,
        default=cfg.get("feedback-timeout", 3.0),
    )
    parser.add_argument(
        "--timeout",
        type=base.positive_float,
        default=cfg.get("timeout", 5.0),
        help="seconds to wait for fresh normal status after both reset commands",
    )
    parser.add_argument(
        "--confirm-drop",
        action="store_true",
        help="confirm both arms are supported and may lose holding torque",
    )
    base.add_execute_argument(parser)
    return parser


def _connection_args(args: argparse.Namespace, channel: str) -> SimpleNamespace:
    return SimpleNamespace(
        interface=args.interface,
        channel=channel,
        firmware=args.firmware,
        feedback_timeout=args.feedback_timeout,
    )


def _wait_reset(robot: Any, channel: str, sent_at: float, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_code: int | None = None
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            last_code = base.value_as_int(status.msg.arm_status)
            timestamp = float(getattr(status, "timestamp", 0.0))
            if timestamp > sent_at and last_code == 0:
                return
        time.sleep(0.02)
    raise TimeoutError(
        f"{channel}: no fresh normal arm status within {timeout:.1f}s "
        f"after reset (last arm_status={last_code})"
    )


def _wait_controller_feedback(robot: Any, channel: str, timeout: float) -> str:
    """Check reachability without requiring motion-output-role joint feedback."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for method, source in (
            ("get_arm_status", "arm status"),
            ("get_joint_angles", "normal joint feedback"),
            ("get_leader_joint_angles", "leader drag feedback"),
        ):
            getter = getattr(robot, method, None)
            message = getter() if callable(getter) else None
            if message is None:
                continue
            stamp = float(getattr(message, "timestamp", 0.0))
            age = time.time() - stamp
            if math.isfinite(age) and stamp > 0 and 0 <= age <= timeout:
                return source
        time.sleep(0.02)
    raise TimeoutError(
        f"{channel}: no fresh status, normal joints or leader drag feedback "
        f"within {timeout:.1f}s; no reset sent to either arm. "
        "Leader mode may suppress normal joint feedback; check USB-CAN "
        "and controller communication if all feedback is absent."
    )


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    channels = [str(args.follower_channel), str(args.leader_channel)]
    if not all(channel.strip() for channel in channels):
        raise ValueError("both CAN channel names must be non-empty")
    if channels[0] == channels[1]:
        raise ValueError("follower and leader CAN channels must be different")
    if args.execute and not args.confirm_drop:
        raise RuntimeError(
            "refusing dual-arm reset: support both arms and add --confirm-drop"
        )

    connected: list[tuple[str, Any]] = []
    try:
        # Preflight both controllers before mutating either one.
        for channel in channels:
            robot, _ = base.connect_robot(_connection_args(args, channel))
            connected.append((channel, robot))
            source = _wait_controller_feedback(robot, channel, args.feedback_timeout)
            print(f"{channel}: CAN feedback ready ({source}).", flush=True)

        if not args.execute:
            print(
                "DRY-RUN: both controllers are reachable; no reset was sent. "
                "Support both arms, then add --confirm-drop --execute.",
                flush=True,
            )
            return 0

        sent_times: dict[str, float] = {}
        for channel, robot in connected:
            sent_times[channel] = time.time()
            robot.reset()
            print(f"{channel}: software reset sent.", flush=True)
        for channel, robot in connected:
            _wait_reset(robot, channel, sent_times[channel], args.timeout)
            print(f"{channel}: fresh normal status confirmed.", flush=True)
        print(
            "Both Piper software emergency-stop states are reset. "
            "Physical emergency-stop buttons, if pressed, must be released manually.",
            flush=True,
        )
        return 0
    finally:
        for _, robot in reversed(connected):
            base.disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
