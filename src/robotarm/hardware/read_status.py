#!/usr/bin/env python3
"""01：持续读取并打印机械臂六个关节的角度反馈。

用法：
  python3 -m robotarm.hardware.read_status
  python3 -m robotarm.hardware.read_status --interval 0.05

这是只读脚本，不会使能或移动机械臂；按 Ctrl+C 退出。
迁移自 piper_ctrl/control_scripts/01_read_status.py。
"""

from __future__ import annotations

import argparse
import math
import time

from ..config import defaults
from ..core.robot.piper import (
    add_connection_arguments,
    connect_robot,
    disconnect_quietly,
    wait_for_feedback,
)


def parse_args() -> argparse.Namespace:
    cfg = defaults("data_acquisition", "read-status")
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser, cfg)
    parser.add_argument(
        "--interval",
        type=float,
        default=cfg["interval"],
        help="printing interval in seconds",
    )
    return parser.parse_args()


def print_joint_status(robot: object) -> None:
    joints = robot.get_joint_angles()
    if joints is None:
        print("Joint feedback: unavailable")
        return

    degrees = [math.degrees(value) for value in joints.msg]
    formatted = "  ".join(
        f"J{index}={angle:8.3f}°"
        for index, angle in enumerate(degrees, start=1)
    )
    print(formatted)


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    robot = None
    try:
        robot, _ = connect_robot(args)
        wait_for_feedback(robot, args.feedback_timeout)
        print("Connected. Press Ctrl+C to stop.")
        while True:
            print_joint_status(robot)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    finally:
        disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
