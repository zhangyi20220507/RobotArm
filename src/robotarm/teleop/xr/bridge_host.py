"""XR 桥接发布进程：保持一条 XRoboToolkit SDK 连接并向本机发布状态。

迁移自 XRoboToolkit-Teleop-Sample-Python/scripts/hardware/xrobotoolkit_xr_bridge.py。
用法：python3 -m robotarm.teleop.xr.bridge_host [--endpoint tcp://127.0.0.1:60062]
"""

from __future__ import annotations

import argparse
import signal
import time


def make_parser() -> argparse.ArgumentParser:
    from ...config import defaults

    cfg = defaults("data_acquisition", "xr-bridge")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=cfg["endpoint"],
        help="local ZeroMQ endpoint",
    )
    parser.add_argument("--rate", type=float, default=cfg["rate"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if not 10.0 <= args.rate <= 500.0:
        raise ValueError("--rate must be in [10, 500] Hz")

    import zmq

    import xrobotoolkit_sdk as xrt

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(args.endpoint)
    period = 1.0 / args.rate

    print(f"XR bridge listening on {args.endpoint}")
    print("Keep this process running while restarting Piper control scripts.")
    xrt.init()
    try:
        last_timestamp = None
        ready_reported = False
        while not stopping:
            started = time.monotonic()
            timestamp = int(xrt.get_time_stamp_ns())
            state = {
                "timestamp_ns": timestamp,
                "left_controller_pose": list(xrt.get_left_controller_pose()),
                "left_grip": float(xrt.get_left_grip()),
                "left_trigger": float(xrt.get_left_trigger()),
                "left_axis_click": bool(xrt.get_left_axis_click()),
                "right_controller_pose": list(xrt.get_right_controller_pose()),
                "right_grip": float(xrt.get_right_grip()),
                "right_trigger": float(xrt.get_right_trigger()),
                "right_axis_click": bool(xrt.get_right_axis_click()),
                "X_button": bool(xrt.get_X_button()),
                "Y_button": bool(xrt.get_Y_button()),
                "A_button": bool(xrt.get_A_button()),
                "B_button": bool(xrt.get_B_button()),
            }
            publisher.send_json(state)
            if timestamp not in (0, last_timestamp) and not ready_reported:
                print("XR controller data ready; teleop clients may now be restarted freely.")
                ready_reported = True
            last_timestamp = timestamp
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        xrt.close()
        publisher.close()
        context.term()
    print("XR bridge stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
