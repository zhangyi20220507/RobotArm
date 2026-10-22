"""piper_raw 数据格式的单点定义。

recorder（写入）、convert（raw→LeRobot 转换）和训练侧的预处理都应引用
本模块，杜绝两边 schema 漂移。
"""

from __future__ import annotations


FORMAT_VERSION = "piper_raw_v1"

# 状态向量：6 关节弧度 + 夹爪宽度（米）。
STATE_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "joint_6", "gripper_width",
]
STATE_UNITS = ["rad"] * 6 + ["m"]

ACTION_SEMANTICS = "commanded MOVE_JS joint targets and gripper target"

# 每帧样本的字段名（写入 frames.npz 的键）。
SAMPLE_FIELDS = (
    "timestamp", "observation_state", "action", "observation_tcp_pose",
    "teleop_delta", "camera_host_timestamp", "camera_sensor_timestamp",
    "camera_frame_number", "camera_frame_age", "camera_skew",
)

# 软件主从采集的可选诊断字段。它们保留在原始 NPZ 中，现有 LeRobot
# 转换仍只导出 observation_state/action，因此不会改变训练侧特征结构。
LEADER_FOLLOWER_SAMPLE_FIELDS = (
    "leader_state",
    "follower_state",
    "following_error",
    "leader_feedback_age",
    "follower_feedback_age",
)

# float32 存储的字段；其余浮点字段按 float64，帧号按 int64。
FLOAT32_FIELDS = frozenset(
    {
        "observation_state",
        "action",
        "observation_tcp_pose",
        "teleop_delta",
        "leader_state",
        "follower_state",
        "following_error",
    }
)


def sample_dtype(field: str):
    """Return the numpy dtype used for *field* in frames.npz."""
    import numpy as np

    if field == "camera_frame_number":
        return np.int64
    if field in FLOAT32_FIELDS:
        return np.float32
    return np.float64


def session_metadata(task: str, setup_id: str, setup_note: str, fps: int,
                    robot_metadata: dict, camera_metadata: dict) -> dict:
    """Build the session.json payload."""
    from datetime import datetime

    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "task": task,
        "setup_id": setup_id,
        "setup_note": setup_note,
        "record_fps": fps,
        "state_names": list(STATE_NAMES),
        "state_units": list(STATE_UNITS),
        "action_semantics": ACTION_SEMANTICS,
        "robot": robot_metadata,
        "cameras": camera_metadata,
    }
