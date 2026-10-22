#!/usr/bin/env python3
"""PICO 手柄实时监视：左右手柄 XYZ、轨迹、原点（只读 XR bridge）。

数据来自 xr-bridge 发布的 XR 状态，不连接、不使能、不驱动机械臂。

两个坐标系（图窗左右各一个）：
  * 房间系（PICO/OpenXR）：头显应用的跟踪坐标系，原点由 recenter 决定（通常在
    开机站位附近的地面）。+X = 你的右手方向，+Y = 竖直向上，+Z = 你的背后方向
    （朝机械臂为负）。手柄位置原样发布在这个系里。
  * 机器人基座系（Piper）：机械臂底座的坐标系，原点在底座。+X = 机械臂伸出去的
    方向，+Y = 机器人的左手边，+Z = 竖直向上。遥操时程序把「手柄在房间系里的
    位移」经固定矩阵 + --xr-yaw-deg + --xr-mirror 换算成基座系位移再下发给机械臂。
    注意两个系的"上"不是同一个轴名：房间系的 Y 对应基座系的 Z。

画法：每个坐标系一张 XY 二维图（房间系是正视：横=右、纵=上；基座系是俯视：
横=前、纵=左），Z 用右边的细长一维标尺表示（房间系 Z = 前后，基座系 机Z = 高度）。
基座系画的是相对参考点（默认开局位置）的位移，因为不接机械臂时不知道法兰的绝对位置。

用法：
    conda activate robotarm
    python scripts/xr_monitor.py --yaw 0 --mirror lr    # 终端读数 + 图窗
    python scripts/xr_monitor.py --no-plot              # 无显示环境：读数 + ASCII 视图

图窗按 r 重置参考点、c 清空轨迹、q 退出。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robotarm.teleop.xr.geometry import mapping_matrix, valid_xr_pose  # noqa: E402

HANDS = ("left", "right")

CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "AR PL UMing CN", "Droid Sans Fallback",
)

PANELS = (
    ("room", "房间系 XY（横=你右手 X，纵=上 Y）"),
    ("room_z", "房间系 Z\n(后 +)"),
    ("robot", "基座系 XY（横=机X 前，纵=机Y 左）"),
    ("robot_z", "基座系 机Z\n(上 +)"),
)
# Z 标尺里左右手各占一条竖线，避免两个标记叠在一起
Z_MARKER_X = {"left": -0.35, "right": 0.35}


def configure_cjk_font() -> bool:
    """Pick an installed CJK font so titles render; returns False if none found."""
    from matplotlib import font_manager, rcParams
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return True
    return False


def start_bridge_reader(endpoint: str):
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(endpoint)
    latest = {"state": None, "stamp": 0.0, "rate": 0.0}

    def loop():
        last = time.monotonic()
        while True:
            try:
                state = socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue
            except Exception:
                return  # 退出时 socket 关闭
            now = time.monotonic()
            latest["rate"] = 0.7 * latest["rate"] + 0.3 / max(now - last, 1e-6)
            last = now
            latest["state"] = state
            latest["stamp"] = now

    threading.Thread(target=loop, daemon=True).start()
    return latest, socket


def poses_of(state):
    """Return {hand: 7-vector} for valid poses in the latest bridge frame."""
    out = {}
    if not state:
        return out
    for hand in HANDS:
        pose = np.asarray(state.get(f"{hand}_controller_pose", [0.0] * 7), dtype=float)
        if valid_xr_pose(pose):
            out[hand] = pose
    return out


def render_terminal(poses, reference, mapping, scale, latest, elapsed, plot_ok):
    lines = [
        f"XR 实时监视  t={elapsed:6.1f}s  {latest['rate']:5.1f} Hz   房间系原点 (0,0,0) 由 recenter 决定；"
        f"基座系显示相对参考点（开局位置）的位移",
        f"  {'手柄':<14s}{'X(m)':>9s}{'Y(m)':>9s}{'Z(m)':>9s}"
        f"{'ΔX(mm)':>10s}{'ΔY(mm)':>10s}{'ΔZ(mm)':>10s}   坐标系",
    ]
    for hand, label in (("left", "左手柄"), ("right", "右手柄")):
        pose = poses.get(hand)
        if pose is None:
            lines.append(f"  {label:<14s}{'—':>9s}{'—':>9s}{'—':>9s}"
                         f"{'—':>10s}{'—':>10s}{'—':>10s}   房间系（无数据）")
            continue
        p = pose[:3]
        ref = reference.get(hand)
        delta = (p - ref) if ref is not None else np.zeros(3)
        lines.append(
            f"  {label:<14s}{p[0]:>9.3f}{p[1]:>9.3f}{p[2]:>9.3f}"
            f"{delta[0] * 1000:>10.0f}{delta[1] * 1000:>10.0f}{delta[2] * 1000:>10.0f}"
            f"   房间系 X右/Y上/Z后"
        )
        if ref is not None:
            mapped = mapping @ delta * scale
            lines.append(
                f"  {label}→机器人  机X{mapped[0] * 1000:>+8.0f}  "
                f"机Y{mapped[1] * 1000:>+8.0f}  机Z{mapped[2] * 1000:>+8.0f} mm"
                f"    基座系 机X前/机Y左/机Z上"
            )
    lines.append("  " + ("图窗：r 重置参考点 / c 清轨迹 / q 退出"
                         if plot_ok else "Ctrl+C 退出（--no-plot：下面是 ASCII X–Y 视图）"))
    return lines


def ascii_xy_view(trails, width=60, height=15):
    """X–Y 平面视图（房间系：横=右、纵=上；Z 见上表数值）。"""
    points = [p for trail in trails.values() for p in trail] or [np.zeros(3)]
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    span_x = max(max(xs) - min(xs), 0.3)
    span_y = max(max(ys) - min(ys), 0.3)
    grid = [[" "] * width for _ in range(height)]

    def cell(p):
        col = int(round((p[0] - min(xs)) / span_x * (width - 1)))
        row = int(round((p[1] - min(ys)) / span_y * (height - 1)))
        return max(0, min(width - 1, col)), max(0, min(height - 1, row))

    for trail in trails.values():
        for p in trail:
            col, row = cell(p)
            if grid[row][col] == " ":
                grid[row][col] = "."
    col, row = cell(np.zeros(3))
    grid[row][col] = "+"
    for hand, char in (("left", "L"), ("right", "R")):
        if trails.get(hand):
            col, row = cell(trails[hand][-1])
            grid[row][col] = char
    header = (f"  ASCII X–Y 视图（横轴 X（你右）：{min(xs):+.2f}..{max(xs):+.2f} m，"
              f"纵轴 Y（上）：{min(ys):+.2f}..{max(ys):+.2f} m）  + = 原点  L/R = 当前手柄")
    return [header] + ["  |" + "".join(row) + "|" for row in reversed(grid)]


def build_figure():
    """XY 二维图 + Z 一维标尺；左 = 房间系，右 = 机器人基座系。"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 5.4))
    fig.suptitle("PICO 手柄监视　左：房间系（横=你右 X，纵=上 Y，Z=前后）　"
                 "右：机器人基座系（横=机X 前，纵=机Y 左，机Z=高度，相对参考点）")
    grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 0.30, 1.0, 0.30))
    artists = {}
    for (key, title), cell in zip(PANELS, (grid[0, i] for i in range(4))):
        ax = fig.add_subplot(cell)
        ax.set_title(title, fontsize=9)
        artists[key] = {"ax": ax}
        if key.endswith("_z"):
            ax.set_xlim(-1, 1)
            ax.set_xticks([])
            ax.axhline(0.0, color="black", lw=1.0, ls="--")
            ax.text(-0.95, 0.0, "0", fontsize=8, va="center", ha="left", color="black")
            for hand in HANDS:
                marker = ax.plot([Z_MARKER_X[hand]], [0.0], marker="o", ms=9,
                                 ls="none", label=hand)[0]
                text = ax.text(Z_MARKER_X[hand] * 1.35, 0.0, "", fontsize=8,
                               va="center", color=marker.get_color())
                artists[key][hand] = (marker, text)
            ax.legend(loc="upper center", fontsize=7, ncol=2)
        else:
            for hand in HANDS:
                line = ax.plot([], [], lw=1.4, label=hand)[0]
                dot = ax.plot([], [], marker="o", ms=6, ls="none")[0]
                artists[key][hand] = (line, dot)
            ax.plot([0], [0], marker="+", color="black", ms=13, mew=1.6,
                    zorder=5, label="origin")
            ax.grid(alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper right", fontsize=7)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.84, bottom=0.09,
                        wspace=0.42)
    return fig, artists


def update_plot(fig, artists, trails, mapping, scale, reference):
    def room_points(hand):
        return np.array([p for _, p in trails[hand]]) if trails[hand] else np.zeros((0, 3))

    def robot_points(hand):
        points = room_points(hand)
        ref = reference.get(hand)
        if len(points) == 0 or ref is None:
            return np.zeros((0, 3))
        return (mapping @ (points - ref).T).T * scale

    for key, source in (("room", room_points), ("robot", robot_points)):
        ax = artists[key]["ax"]
        collected = []
        for hand in HANDS:
            points = source(hand)
            line, dot = artists[key][hand]
            line.set_data(points[:, 0], points[:, 1])
            dot.set_data(points[-1:, 0], points[-1:, 1])
            if len(points):
                collected.append(points[:, :2])
        if collected:
            pts = np.concatenate(collected)
            lo = np.minimum(pts.min(axis=0), 0.0)
            hi = np.maximum(pts.max(axis=0), 0.0)
            pad = 0.15 * max(float(np.max(hi - lo)), 0.2)
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)

    for key, source in (("room_z", room_points), ("robot_z", robot_points)):
        ax = artists[key]["ax"]
        values = []
        for hand in HANDS:
            points = source(hand)
            value = float(points[-1, 2]) if len(points) else 0.0
            marker, text = artists[key][hand]
            marker.set_data([Z_MARKER_X[hand]], [value])
            text.set_position((Z_MARKER_X[hand] * 1.35, value))
            text.set_text(f"{value * 1000:+.0f}mm")
            values.append(abs(value))
        span = max(max(values) * 1.3, 0.05) if values else 0.05
        ax.set_ylim(-span, span)
    fig.canvas.draw_idle()


def on_key(event, latest, trails, reference):
    if event.key == "r":
        for hand, pose in poses_of(latest["state"]).items():
            reference[hand] = pose[:3].copy()
        print("\n参考点已重置")
    elif event.key == "c":
        for hand in HANDS:
            trails[hand].clear()
        print("\n轨迹已清空")
    elif event.key == "q":
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:60062")
    parser.add_argument("--yaw", type=float, default=0.0, help="与遥操一致的 --xr-yaw-deg")
    parser.add_argument("--mirror", choices=("none", "lr", "fb"), default="none")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--trail", type=float, default=60.0, help="轨迹保留秒数")
    parser.add_argument("--no-plot", action="store_true", help="不打开图窗，用 ASCII X–Y 视图")
    args = parser.parse_args()

    mapping = mapping_matrix(args.yaw, args.mirror)
    latest, socket = start_bridge_reader(args.endpoint)
    print(f"等待 XR 数据（{args.endpoint}）…  Ctrl+C 退出", flush=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not poses_of(latest["state"]):
        time.sleep(0.05)
    if not poses_of(latest["state"]):
        print("没有收到有效手柄姿态：确认 xr-bridge 在运行、头显应用已开 Send、手柄已唤醒")
        socket.close(linger=0)
        return 1

    trails = {hand: [] for hand in HANDS}
    reference = {}
    started = time.monotonic()
    printed = 0
    fig = artists = None
    plot_ok = False

    if not args.no_plot:
        try:
            import matplotlib
            if not matplotlib.get_backend().lower().startswith("agg"):
                import matplotlib.pyplot as plt
                if not configure_cjk_font():
                    print("未找到中文字体，图标题可能显示为方框（数值不受影响）")
                plt.ion()
                fig, artists = build_figure()
                fig.canvas.mpl_connect(
                    "key_press_event",
                    lambda event: on_key(event, latest, trails, reference))
                plot_ok = True
                print("图窗已打开：左=房间系，右=机器人基座系；r 重置参考点 / c 清轨迹 / q 退出")
        except Exception as exc:
            print(f"打开图窗失败（{exc}），改用 ASCII 视图")
    print()
    return run_loop(latest, socket, mapping, args, trails, reference, started,
                    plot_ok, fig, artists, printed)


def run_loop(latest, socket, mapping, args, trails, reference, started,
             plot_ok, fig, artists, printed):
    next_display = 0.0
    next_plot = 0.0
    try:
        while True:
            now = time.monotonic()
            poses = poses_of(latest["state"])
            fresh = latest["state"] is not None and now - latest["stamp"] < 0.5
            for hand, pose in poses.items():
                reference.setdefault(hand, pose[:3].copy())
                trails[hand].append((now, pose[:3].copy()))
                cutoff = now - args.trail
                while trails[hand] and trails[hand][0][0] < cutoff:
                    trails[hand].pop(0)

            if now >= next_display:
                next_display = now + 0.2
                lines = render_terminal(poses, reference, mapping, args.scale,
                                        latest, now - started, plot_ok)
                if not plot_ok:
                    lines += ascii_xy_view({h: [p for _, p in t] for h, t in trails.items()})
                if not fresh:
                    lines[0] = "（XR 数据已过期，检查 bridge / 头显） " + lines[0]
                if printed:
                    print(f"\033[{printed}A", end="")
                for line in lines:
                    print("\033[2K" + line)
                print("\033[0J", end="")
                printed = max(printed, len(lines))

            if plot_ok and now >= next_plot:
                import matplotlib.pyplot as plt
                next_plot = now + 0.07
                update_plot(fig, artists, trails, mapping, args.scale, reference)
                plt.pause(0.001)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if printed:
            print(f"\033[{printed}A", end="")
            for _ in range(printed):
                print("\033[2K")
            print("\033[0J", end="")
        socket.close(linger=0)
        print("监视结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
