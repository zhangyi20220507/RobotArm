"""订阅持久 XR 桥接进程的 ZeroMQ 客户端，接口与 xrobotoolkit_sdk 对齐。"""

from __future__ import annotations


class XrBridgeClient:
    """Module-like XR source backed by the persistent local bridge."""

    def __init__(self, endpoint: str):
        import zmq

        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.state = {
            "timestamp_ns": 0,
            "left_controller_pose": [0.0] * 7,
            "left_grip": 0.0,
            "left_trigger": 0.0,
            "left_axis_click": False,
            "right_controller_pose": [0.0] * 7,
            "right_grip": 0.0,
            "right_trigger": 0.0,
            "right_axis_click": False,
            "X_button": False,
            "Y_button": False,
            "A_button": False,
            "B_button": False,
        }

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.socket.close()
        self.context.term()

    def _refresh(self) -> None:
        while True:
            try:
                self.state = self.socket.recv_json(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                return

    def snapshot(self):
        """Read one complete published frame for a control tick."""
        self._refresh()
        return dict(self.state)

    def get_time_stamp_ns(self):
        self._refresh()
        return self.state["timestamp_ns"]

    def get_left_controller_pose(self):
        self._refresh()
        return self.state["left_controller_pose"]

    def get_left_grip(self):
        self._refresh()
        return self.state["left_grip"]

    def get_left_trigger(self):
        self._refresh()
        return self.state["left_trigger"]

    def get_left_axis_click(self):
        self._refresh()
        return self.state["left_axis_click"]

    def get_right_controller_pose(self):
        self._refresh()
        return self.state["right_controller_pose"]

    def get_right_grip(self):
        self._refresh()
        return self.state["right_grip"]

    def get_right_trigger(self):
        self._refresh()
        return self.state["right_trigger"]

    def get_right_axis_click(self):
        self._refresh()
        return self.state["right_axis_click"]

    def get_X_button(self):
        self._refresh()
        return self.state["X_button"]

    def get_Y_button(self):
        self._refresh()
        return self.state["Y_button"]

    def get_A_button(self):
        self._refresh()
        return self.state["A_button"]

    def get_B_button(self):
        self._refresh()
        return self.state["B_button"]
