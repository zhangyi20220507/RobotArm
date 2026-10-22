#!/usr/bin/env python3
"""PICO 遥操坐标系验证（只读：只订阅 XR bridge，不连接、不驱动机械臂）。

用法（先启动 xr-bridge，再戴好头显、握好手柄）：

    conda activate robotarm
    robotarm data-acquisition xr-bridge -- --endpoint tcp://127.0.0.1:60062 --rate 100   # 终端 1
    python scripts/check_xr_frame.py --hand right --yaw 0 --mirror none                   # 终端 2

只订阅 bridge 发布的 XR 状态，不连接、不使能、不驱动机械臂。

按提示做 4 个动作，每个动作前后各按一次回车。脚本会打印
  * 动作在【PICO 坐标系】里的位移分量（并检查它是否符合 OpenXR 约定）
  * 同一动作在【机器人基座坐标系】里的位移（用程序里同一套映射矩阵）
  * 所有 yaw x mirror 组合的结果表，便于挑出符合你直觉的那一组
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

# 允许从任意工作目录运行：优先使用仓库内的 src/，保证与运行时同一套映射实现。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robotarm.teleop.xr.geometry import mapping_matrix, valid_xr_pose, xr_rotation

XR_LABELS = ("XR右", "XR上", "XR后")       # OpenXR 约定：+X 右、+Y 上、+Z 后（相对 recenter 朝向）
ROBOT_LABELS = ("机X", "机Y", "机Z")        # Piper 基座：+X 前(伸出方向)、+Y 左、+Z 上


def start_bridge_reader(endpoint: str):
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(endpoint)
    latest = {"state": None, "stamp": 0, "count": 0}

    def loop():
        while True:
            try:
                state = socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            except Exception:
                return  # socket closed at shutdown
            latest["state"] = state
            latest["stamp"] = time.monotonic()
            latest["count"] += 1

    threading.Thread(target=loop, daemon=True).start()
    return latest, socket


def wait_fresh(latest, timeout=10.0, quiet=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = latest["state"]
        if state is not None and time.monotonic() - latest["stamp"] < 0.5:
            pose = np.asarray(state[f"{HAND}_controller_pose"], dtype=float)
            if valid_xr_pose(pose):
                return pose
        time.sleep(0.05)
    if not quiet:
        print("没有收到有效手柄数据：确认 xr-bridge 在运行、头显应用已开 Send、手柄已唤醒")
    return None


def ask(prompt):
    """input() that survives closed/piped stdin."""
    try:
        return input(prompt)
    except EOFError:
        print()
        return None


def describe(delta, labels, threshold=0.004):
    parts = [
        f"{labels[i]}{delta[i] * 1000:+.0f}mm"
        for i in range(3)
        if abs(delta[i]) > threshold
    ]
    return " ".join(parts) or "（几乎没动）"


def record(label, instruction):
    print(f"\n[{label}] {instruction}")
    if ask("  → 摆好起始姿势后按回车记录起点…") is None:
        return None
    start = wait_fresh(latest)
    if start is None:
        return None
    if ask("  → 做完动作、保持住，按回车记录终点…") is None:
        return None
    end = wait_fresh(latest)
    if end is None:
        return None
    return start, end


def main():
    global HAND, latest
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--yaw", type=float, default=0.0, help="当前使用的 --xr-yaw-deg")
    parser.add_argument("--mirror", choices=("none", "lr", "fb"), default="none",
                        help="当前使用的 --xr-mirror")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:60062")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    HAND = args.hand

    latest, socket = start_bridge_reader(args.endpoint)
    print(f"等待 {HAND} 手柄数据（{args.endpoint}）…")
    pose = wait_fresh(latest)
    if pose is None:
        return 1
    print("已连接。位置单位：米，姿态为四元数 [x,y,z,qx,qy,qz,qw]（XRoboToolkit 约定）")
    print(f"当前手柄位置（PICO 坐标系）: {np.round(pose[:3], 4)}")

    mapping = mapping_matrix(args.yaw, args.mirror)
    steps = [
        ("前方", "把*手柄*朝【机械臂方向】平推约 10 cm（尽量别转手腕）"),
        ("右侧", "把*手柄*朝【你自己的右手方向】平移约 10 cm"),
        ("上抬", "把*手柄*向上抬约 10 cm"),
    ]
    results = {}
    for key, instruction in steps:
        sample = record(key, instruction)
        if sample is None:
            return 1
        start, end = sample
        delta_xr = end[:3] - start[:3]
        delta_robot = mapping @ delta_xr * args.scale
        results[key] = delta_xr
        print(f"  PICO 坐标系位移: {describe(delta_xr, XR_LABELS)}")
        print(f"  机器人坐标系位移: {describe(delta_robot, ROBOT_LABELS)}"
              f"   (yaw={args.yaw:g}, mirror={args.mirror}, scale={args.scale:g})")

    # 手腕旋转（可选）
    print("\n[朝向] 把手柄绕【竖直轴】向左转约 30°；不测朝向就直接回车跳过")
    answer = ask("  按回车开始，或输入 s 跳过: ")
    if answer is not None and answer.strip().lower() != "s":
        start = wait_fresh(latest)
        if ask("  → 转好后按回车…") is None:
            start = None
        end = wait_fresh(latest) if start is not None else None
        if start is not None and end is not None:
            delta_xr_matrix = xr_rotation(end).as_matrix() @ xr_rotation(start).as_matrix().T
            delta_robot = mapping @ delta_xr_matrix @ mapping.T
            from scipy.spatial.transform import Rotation
            euler = Rotation.from_matrix(delta_robot).as_euler("xyz", degrees=True)
            magnitude = math.degrees((xr_rotation(end) * xr_rotation(start).inv()).magnitude())
            print(f"  PICO 坐标系转动 {magnitude:.1f}° → 机器人法兰转动 "
                  f"roll/pitch/yaw = {np.round(euler, 1)}°（需要 --orientation 才生效）")

    # 候选映射表
    print("\n=== yaw / mirror 候选表（用你刚才那次『前方』动作的位移）===")
    forward = results.get("前方")
    if forward is not None:
        print(f"{'yaw':>6s} {'mirror':>7s}   机器人坐标系位移")
        for yaw in (0.0, 90.0, 180.0, 270.0):
            for mirror in ("none", "lr", "fb"):
                delta = mapping_matrix(yaw, mirror) @ forward * args.scale
                print(f"{yaw:6.0f} {mirror:>7s}   {describe(delta, ROBOT_LABELS)}")
    print("\n判定方法：机械臂基座 +X 是臂伸出去的方向、+Y 是机器人的左手边、+Z 向上。")
    print("选那一组让『手朝机械臂推 → 机X 为正』『手抬高 → 机Z 为正』"
          "『手往你右边 → 机Y 为负』的结果；\n"
          "把选定的 yaw/mirror 填进 xr.yaml 或命令行（--xr-yaw-deg / --xr-mirror）。")

    # 端到端复核提示
    print("\n端到端复核（可选）：单臂 dry-run 会每秒打印 hand_delta / requested_delta，")
    print("对比两者的方向即可确认整条链路的映射是否和上面一致：")
    print("  robotarm data-acquisition piper-xr -- --arms 1 --xr-bridge --channel can_follower1 --firmware v189")
    socket.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
