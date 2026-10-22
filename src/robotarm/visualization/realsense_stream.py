#!/usr/bin/env python3
"""实时显示所有已连接 RealSense 相机的彩色+深度画面。

用法：python3 -m robotarm.visualization.realsense_stream
按 q 退出。迁移自 rs_cam_streaming.py（原依赖 xrobotoolkit_teleop，
现直接使用 pyrealsense2）。

USB 2.x 口上彩色+深度双流带宽不足（帧无法到达），此时自动仅开
彩色流；`--depth` 可强制尝试双流（建议插 USB 3.0 口），
`--color-only` 可在任何 USB 连接下强制只显示彩色图像。
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serials",
        nargs="*",
        default=None,
        help="only stream these camera serials (default: all connected)",
    )
    stream_mode = parser.add_mutually_exclusive_group()
    stream_mode.add_argument(
        "--depth",
        action="store_true",
        help="force color+depth streaming even on USB 2.x",
    )
    stream_mode.add_argument(
        "--color-only",
        action="store_true",
        help="stream only the RGB color image",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="display each camera window in fullscreen mode",
    )
    return parser.parse_args()


def main() -> int:
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    args = parse_args()

    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        print("No RealSense devices found.")
        return 1

    selected = {}
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if args.serials is None or serial in args.serials:
            selected[serial] = device
    if not selected:
        print("None of the requested serials are connected.")
        return 1
    print("Streaming cameras: " + ", ".join(selected))

    pipelines: dict[str, rs.pipeline] = {}
    depth_views: dict[str, bool] = {}
    windows: set[str] = set()
    try:
        for serial, device in selected.items():
            name = device.get_info(rs.camera_info.name)
            usb = device.get_info(rs.camera_info.usb_type_descriptor)
            with_depth = not args.color_only and (
                args.depth or usb.startswith(("3", "3."))
            )
            if not with_depth:
                print(
                    f"{serial}: {name} (USB {usb}) — color only; "
                    "color+depth needs USB 3.0 (override with --depth)"
                )
            else:
                print(f"{serial}: {name} (USB {usb}) — color+depth")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if with_depth:
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipeline.start(config)
            pipelines[serial] = pipeline
            depth_views[serial] = with_depth

        # D435 冷启动首帧可能超过 2 秒，超时放宽到 5 秒
        timeout_ms = 5000
        failures = {serial: 0 for serial in pipelines}
        while True:
            for serial, pipeline in pipelines.items():
                try:
                    frameset = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                except RuntimeError:
                    failures[serial] += 1
                    if failures[serial] > 5:
                        print(
                            f"{serial}: no frames after {failures[serial]} "
                            "timeouts — check USB bandwidth/port"
                        )
                    continue
                color = frameset.get_color_frame()
                if not color:
                    continue
                color_image = np.asanyarray(color.get_data())
                if depth_views[serial]:
                    depth = frameset.get_depth_frame()
                    if depth is None:
                        continue
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(
                            np.asanyarray(depth.get_data()), alpha=0.03
                        ),
                        cv2.COLORMAP_JET,
                    )
                    images = np.hstack((color_image, depth_colormap))
                else:
                    images = color_image
                window = f"RealSense - {serial}"
                if window not in windows:
                    mode = cv2.WINDOW_NORMAL if args.fullscreen else cv2.WINDOW_AUTOSIZE
                    cv2.namedWindow(window, mode)
                    if args.fullscreen:
                        cv2.setWindowProperty(
                            window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                        )
                    windows.add(window)
                cv2.imshow(window, images)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for pipeline in pipelines.values():
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
