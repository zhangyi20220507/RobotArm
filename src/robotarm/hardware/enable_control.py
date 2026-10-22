#!/usr/bin/env python3
"""02：统一管理机械臂使能、失能和控制器复位。

用法：
  使能预览：python3 -m robotarm.hardware.enable_control enable
  执行使能：python3 -m robotarm.hardware.enable_control enable --execute
  执行失能：python3 -m robotarm.hardware.enable_control disable --confirm-drop --execute
  执行复位：python3 -m robotarm.hardware.enable_control reset --confirm-drop --execute

失能或复位前必须托住机械臂并添加 --confirm-drop，防止机械臂失去保持力后下落。
迁移自 piper_ctrl/control_scripts/02_enable_control.py。
"""

from __future__ import annotations

import argparse
import time

from ..config import defaults
from ..core.robot.piper import (
    add_connection_arguments,
    add_execute_argument,
    connect_robot,
    disable_arm_and_gripper,
    disconnect_quietly,
    enable_arm_and_gripper,
    value_as_int,
    wait_for_feedback,
    wait_gripper_width,
)


def parse_args() -> argparse.Namespace:
    cfg = defaults("data_acquisition", "enable-control")
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser, cfg)
    add_execute_argument(parser)
    parser.add_argument(
        "action",
        choices=("enable", "disable", "reset"),
        help="enable arm+gripper; disable holding power; reset controller and power off",
    )
    parser.add_argument(
        "--confirm-drop",
        action="store_true",
        help="confirm the arm is supported and may safely lose power",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=cfg["timeout"],
        help="operation timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")

    robot = None
    try:
        robot, _ = connect_robot(args)
        wait_for_feedback(robot, args.feedback_timeout)

        if not args.execute:
            print(f"DRY-RUN: CAN通信正常，将执行 {args.action}。")
            if args.action in ("disable", "reset"):
                print("警告：失能或复位后机械臂可能立即下落。")
            print("确认安全后添加 --execute 才会发送控制指令。")
            return 0

        if args.action in ("disable", "reset") and not args.confirm_drop:
            raise RuntimeError(
                "拒绝执行：请先支撑机械臂并清空工作区域，然后添加 "
                "--confirm-drop。"
            )

        if args.action == "reset":
            robot.reset()
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                status = robot.get_arm_status()
                if status is not None and value_as_int(status.msg.arm_status) == 0:
                    print("机械臂控制器已复位；关节可能已失去保持力。")
                    return 0
                time.sleep(0.05)
            raise TimeoutError(
                f"控制器在 {args.timeout:.1f} 秒内未恢复正常状态。"
            )

        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        gripper_width = wait_gripper_width(gripper, args.feedback_timeout)

        if args.action == "enable":
            enable_arm_and_gripper(
                robot, gripper, gripper_width, timeout=args.timeout
            )
            print("机械臂六个关节和夹爪已成功使能。")
            return 0


        disable_arm_and_gripper(
            robot, gripper, gripper_width, timeout=args.timeout
        )
        print("机械臂六个关节和夹爪已失能。")
        return 0
    except KeyboardInterrupt:
        print("\n操作被用户终止；未自动发送失能指令。")
        return 130
    finally:
        disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
