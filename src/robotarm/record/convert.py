#!/usr/bin/env python3
"""Convert one or more Piper raw sessions into a minimal LeRobot v3 dataset.

Run this module inside the dedicated ``lerobot`` environment. Complete
episodes are converted by default; incomplete episodes require the explicit
``--include-incomplete`` flag.

Representation is selected by ``--space`` and ``--frame``:

- ``--space joint`` (default): state/action are 6 joint angles + gripper width.
- ``--space tcp``: state/action are the end-effector pose [x,y,z,r,p,y] +
  gripper width. The pose is computed with the same MDH forward kinematics
  the driver used during recording, so it matches the raw
  ``observation_tcp_pose`` exactly.
- ``--frame absolute`` (default): values are exported as recorded (or FK'd).
- ``--frame relative``: the action is exported as the per-frame increment
  ``action - state`` (state itself stays absolute). In TCP space the RPY
  components of the increment are wrapped to [-pi, pi].

Teleoperation and capture-diagnostic arrays remain in the unmodified raw
session and are never exposed to the policy.

迁移自 piper_ctrl/control_scripts/10_convert_piper_raw_to_lerobot.py。
用法：python3 -m robotarm.record.convert <session_dir> [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


RAW_FORMAT = "piper_raw_v1"
STATE_WIDTH = 7
ACTION_WIDTH = 7
TCP_WIDTH = 6
TELEOP_WIDTH = 6

# TCP-space feature names: end-effector pose followed by gripper width.
TCP_STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_width"]

# Only causal observations that are available during autonomous deployment belong
# in the LeRobot feature schema.  Teleoperation commands and capture bookkeeping
# remain in the immutable raw NPZ files for diagnostics, but exposing them as
# ``observation.*`` lets policies use them as shortcuts.
MODEL_OBSERVATION_KEYS = ("observation.state", "observation.images.*")
LEROBOT_BOOKKEEPING_KEYS = {
    "timestamp", "frame_index", "episode_index", "index", "task_index"
}


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    metadata: dict[str, Any]
    session: dict[str, Any]
    session_path: Path


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def default_output(session: Path) -> Path:
    # data/raw/piper_raw/<session> -> data/lerobot/<session>
    if session.parent.name == "piper_raw":
        if session.parent.parent.name == "raw":
            return session.parent.parent.parent / "lerobot" / session.name
        # 旧布局 datasets/piper_raw/<session> -> datasets/lerobot/<session>
        return session.parent.parent / "lerobot" / session.name
    return session.parent / f"{session.name}_lerobot"


def default_repo_id(session: Path) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", session.name).strip("-.")
    return f"local/piper-{safe_name or 'dataset'}"


def representation_suffix(space: str, frame: str) -> str:
    """Directory/repo suffix for non-default representations."""
    if space == "joint" and frame == "absolute":
        return ""
    return "_" + "_".join(
        part for part in (space if space != "joint" else "", frame if frame != "absolute" else "")
        if part
    )


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle difference to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_tcp_fk() -> Callable[[list[float]], list[float]]:
    """Return an offline FK bound to the vendored SDK's Piper MDH table.

    The driver used during recording computes ``robot.fk`` from the same
    ``fk_from_mdh``/``get_mdh`` helpers, so TCP-space conversions are exactly
    consistent with the raw ``observation_tcp_pose`` samples.
    """
    from ..paths import PYAGXARM_ROOT

    if not PYAGXARM_ROOT.is_dir():
        raise RuntimeError(f"Local pyAgxArm source not found: {PYAGXARM_ROOT}")
    sdk_path = str(PYAGXARM_ROOT)
    # Same precedence fix as core.robot.piper._load_sdk: force the vendored
    # SDK first so an unrelated installed copy cannot win.
    while sdk_path in sys.path:
        sys.path.remove(sdk_path)
    sys.path.insert(0, sdk_path)
    try:
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import pyAgxArm MDH kinematics: {exc}\n"
            f"Current Python: {sys.executable}"
        ) from exc
    mdh = get_mdh("piper")
    return lambda joints: fk_from_mdh(mdh, list(joints))


def discover_episodes(
    session_path: Path,
    include_incomplete: bool,
) -> tuple[dict[str, Any], list[EpisodeSource], list[str]]:
    session = read_json(session_path / "session.json")
    if session.get("format_version") != RAW_FORMAT:
        raise RuntimeError(
            f"unsupported source format {session.get('format_version')!r}; "
            f"expected {RAW_FORMAT!r}"
        )
    if session.get("action_semantics") != (
        "commanded MOVE_JS joint targets and gripper target"
    ):
        raise RuntimeError(
            "ACT conversion requires absolute MOVE_JS joint targets plus a "
            "gripper target; unexpected action_semantics="
            f"{session.get('action_semantics')!r}"
        )
    expected_units = ["rad"] * 6 + ["m"]
    if session.get("state_units") != expected_units:
        raise RuntimeError(
            f"state_units must be {expected_units}, got {session.get('state_units')!r}"
        )
    episodes_dir = session_path / "episodes"
    if not episodes_dir.is_dir():
        raise RuntimeError(f"missing episodes directory: {episodes_dir}")

    selected: list[EpisodeSource] = []
    skipped: list[str] = []
    for path in sorted(item for item in episodes_dir.iterdir() if item.is_dir()):
        metadata = read_json(path / "episode.json")
        status = metadata.get("status")
        frames = int(metadata.get("frames", 0))
        if frames <= 0:
            skipped.append(f"{path.name}: no frames")
            continue
        if status != "complete" and not include_incomplete:
            skipped.append(f"{path.name}: status={status}")
            continue
        selected.append(
            EpisodeSource(
                path=path,
                metadata=metadata,
                session=session,
                session_path=session_path,
            )
        )
    if not selected:
        raise RuntimeError("no eligible episodes found")
    return session, selected, skipped


def select_camera_roles(
    recorded: list[str], requested: list[str] | None
) -> list[str]:
    """Return the camera roles to convert, in recorded order.

    ``requested`` is the ``--camera-role`` selection: ``None`` (or empty) keeps
    every recorded role. Asking for a role the session never recorded, or for
    the same role twice, is a mistake worth failing on rather than silently
    ignoring.
    """
    if not requested:
        return list(recorded)
    duplicates = sorted({role for role in requested if requested.count(role) > 1})
    if duplicates:
        raise RuntimeError(
            "duplicate --camera-role: " + ", ".join(duplicates)
        )
    unknown = [role for role in requested if role not in recorded]
    if unknown:
        raise RuntimeError(
            "requested camera role(s) not recorded in this session: "
            + ", ".join(unknown)
            + f"; available: {', '.join(recorded)}"
        )
    return [role for role in recorded if role in requested]


def validate_array(name: str, array: Any, rows: int, width: int | None) -> None:
    import numpy as np

    expected_shape = (rows,) if width is None else (rows, width)
    if array.shape != expected_shape:
        raise RuntimeError(
            f"{name} has shape {array.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise RuntimeError(f"{name} is not numeric: {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains NaN or infinity")


def validate_episode(
    episode: EpisodeSource,
    session: dict[str, Any],
    expected_roles: list[str],
) -> None:
    import cv2
    import numpy as np

    rows = int(episode.metadata["frames"])
    roles = list(episode.metadata.get("camera_roles", []))
    if roles != expected_roles:
        raise RuntimeError(
            f"{episode.path.name}: camera roles {roles} differ from "
            f"dataset roles {expected_roles}"
        )
    archive_path = episode.path / "frames.npz"
    try:
        archive = np.load(archive_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"cannot load {archive_path}: {exc}") from exc
    required = {
        "timestamp": None,
        "observation_state": STATE_WIDTH,
        "action": ACTION_WIDTH,
        "observation_tcp_pose": TCP_WIDTH,
        "teleop_delta": TELEOP_WIDTH,
        "camera_host_timestamp": len(expected_roles),
        "camera_sensor_timestamp": len(expected_roles),
        "camera_frame_number": len(expected_roles),
        "camera_frame_age": len(expected_roles),
        "camera_skew": None,
    }
    with archive:
        for key, width in required.items():
            if key not in archive:
                raise RuntimeError(f"{episode.path.name}: missing array {key}")
            validate_array(
                f"{episode.path.name}:{key}", archive[key], rows, width
            )
        timestamps = archive["timestamp"]
        if rows > 1 and not np.all(np.diff(timestamps) > 0):
            raise RuntimeError(
                f"{episode.path.name}: timestamps are not strictly increasing"
            )

    fps = float(session["record_fps"])
    for role in expected_roles:
        video_path = episode.path / f"{role}.mp4"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        try:
            video_rows = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        stream = session["cameras"][role]["stream"]
        if video_rows != rows:
            raise RuntimeError(
                f"{episode.path.name}:{role}.mp4 has {video_rows} frames, "
                f"expected {rows}"
            )
        if not math.isclose(video_fps, fps, abs_tol=0.1):
            raise RuntimeError(
                f"{episode.path.name}:{role}.mp4 fps={video_fps}, expected {fps}"
            )
        if (width, height) != (int(stream["width"]), int(stream["height"])):
            raise RuntimeError(
                f"{episode.path.name}:{role}.mp4 is {width}x{height}, "
                f"expected {stream['width']}x{stream['height']}"
            )


def vector_feature(width: int, names: list[str], dtype: str = "float32") -> dict:
    return {"dtype": dtype, "shape": (width,), "names": names}


def build_features(
    session: dict[str, Any], camera_roles: list[str], space: str = "joint"
) -> dict[str, dict[str, Any]]:
    if space == "tcp":
        state_names = list(TCP_STATE_NAMES)
    else:
        state_names = list(session.get("state_names", []))
        if len(state_names) != STATE_WIDTH:
            raise RuntimeError(
                f"session state_names has {len(state_names)} values, expected 7"
            )
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_WIDTH,),
            "names": {"motors": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_WIDTH,),
            "names": {"motors": state_names},
        },
    }
    for role in camera_roles:
        stream = session["cameras"][role]["stream"]
        features[f"observation.images.{role}"] = {
            "dtype": "video",
            "shape": (int(stream["height"]), int(stream["width"]), 3),
            "names": ["height", "width", "channel"],
        }
    return features


def validate_act_output_schema(output: Path, camera_roles: list[str]) -> None:
    """Reject conversion output that exposes extra observations to ACT."""
    info = read_json(output / "meta" / "info.json")
    actual = set(info.get("features", {}))
    expected_training = {
        "observation.state",
        "action",
        *(f"observation.images.{role}" for role in camera_roles),
    }
    expected = expected_training | LEROBOT_BOOKKEEPING_KEYS
    if actual != expected:
        raise RuntimeError(
            "ACT output schema mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    observation_keys = sorted(key for key in actual if key.startswith("observation."))
    expected_observations = sorted(expected_training - {"action"})
    if observation_keys != expected_observations:
        raise RuntimeError(
            f"ACT observations are {observation_keys}, expected {expected_observations}"
        )


def validate_fk_consistency(
    episode: EpisodeSource, fk: Callable[[list[float]], list[float]], rows: int
) -> None:
    """Check offline FK against the poses recorded alongside the joints."""
    import numpy as np

    with np.load(episode.path / "frames.npz", allow_pickle=False) as data:
        joints = data["observation_state"][:, :6]
        recorded = data["observation_tcp_pose"]
    for index in {0, rows // 2, rows - 1}:
        computed = np.asarray(fk(joints[index].tolist()), dtype=np.float64)
        if not np.allclose(
            computed, recorded[index], rtol=0.0, atol=1e-3
        ):
            raise RuntimeError(
                f"{episode.path.name}: offline FK disagrees with recorded "
                f"observation_tcp_pose at frame {index} "
                f"(max diff {np.max(np.abs(computed - recorded[index])):.4f}); "
                "refusing to build TCP-space actions on inconsistent data"
            )


def compute_action_array(
    data: Any, rows: int, space: str, frame: str, fk: Callable[[list[float]], list[float]]
) -> Any:
    """Build the action array in the requested space and frame."""
    import numpy as np

    if space == "tcp" and fk is None:
        raise RuntimeError("internal error: TCP space requires an FK function")
    state = data["observation_state"]
    action = np.array(data["action"], dtype=np.float64)
    if space == "tcp":
        # State pose: use the recorded TCP poses (already FK of state joints).
        state_pose = np.asarray(data["observation_tcp_pose"], dtype=np.float64)
        # Action pose: FK of the commanded joint targets. This is the same
        # MDH chain the driver used at recording time.
        action_pose = np.asarray(
            [fk(row[:6].tolist()) for row in data["action"]], dtype=np.float64
        )
        state = np.concatenate(
            [state_pose, state[:, 6:7].astype(np.float64)], axis=1
        )
        action = np.concatenate(
            [action_pose, action[:, 6:7]], axis=1
        )
    if frame == "relative":
        action = action - state.astype(np.float64)
        if space == "tcp":
            # RPY increments are wrapped to [-pi, pi); positions and the
            # gripper are plain differences.
            action[:, 3:6] = np.vectorize(wrap_to_pi)(action[:, 3:6])
    if rows and action.shape[0] != rows:
        raise RuntimeError("internal error: action row count mismatch")
    return action.astype(np.float32)


def convert_episode(
    dataset: Any,
    episode: EpisodeSource,
    camera_roles: list[str],
    space: str = "joint",
    frame: str = "absolute",
    fk: Callable[[list[float]], list[float]] | None = None,
) -> int:
    import cv2
    import numpy as np

    rows = int(episode.metadata["frames"])
    captures = {
        role: cv2.VideoCapture(str(episode.path / f"{role}.mp4"))
        for role in camera_roles
    }
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError(f"failed to open a video in {episode.path}")

    try:
        with np.load(episode.path / "frames.npz", allow_pickle=False) as data:
            action_array = compute_action_array(data, rows, space, frame, fk)
            if space == "joint":
                state_array = data["observation_state"]
            else:
                state_array = np.concatenate(
                    [
                        np.asarray(data["observation_tcp_pose"], dtype=np.float64),
                        data["observation_state"][:, 6:7].astype(np.float64),
                    ],
                    axis=1,
                ).astype(np.float32)
            for index in range(rows):
                frame_dict: dict[str, Any] = {
                    "observation.state": state_array[index],
                    "action": action_array[index],
                    "task": str(episode.metadata["task"]),
                }
                for role in camera_roles:
                    ok, image_bgr = captures[role].read()
                    if not ok or image_bgr is None:
                        raise RuntimeError(
                            f"{episode.path.name}:{role}.mp4 ended at frame {index}"
                        )
                    frame_dict[f"observation.images.{role}"] = np.ascontiguousarray(
                        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    )
                dataset.add_frame(frame_dict)
                if (index + 1) % 100 == 0 or index + 1 == rows:
                    print(
                        f"  {episode.path.name}: {index + 1}/{rows} frames",
                        flush=True,
                    )
        dataset.save_episode()
    finally:
        for capture in captures.values():
            capture.release()
    return rows


def parse_args() -> argparse.Namespace:
    from ..config import defaults

    cfg = defaults("data_acquisition", "convert")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions", type=Path, nargs="+", help="piper_raw_v1 session directories"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="local LeRobot repo id (default: local/piper-<session>)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also convert episodes whose status is not complete",
    )
    parser.add_argument(
        "--camera-role",
        action="append",
        default=None,
        metavar="ROLE",
        help=(
            "convert only this camera role (repeatable). Defaults to every "
            "role recorded in the session. The source session is still "
            "validated in full; only the written features change."
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing output directory before conversion",
    )
    output_mode.add_argument(
        "--append",
        action="store_true",
        help="append new sessions to an existing compatible LeRobot dataset",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=cfg["image-writer-threads"],
        help="LeRobot temporary image writer threads",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=cfg["encoder-threads"],
        help="video encoder threads",
    )
    parser.add_argument(
        "--space",
        choices=("joint", "tcp"),
        default=cfg["space"],
        help=(
            "representation for observation.state/action: 'joint' = 6 joint "
            "angles + gripper width (default); 'tcp' = end-effector pose "
            "[x,y,z,r,p,y] + gripper width, computed with the recording-time "
            "MDH forward kinematics"
        ),
    )
    parser.add_argument(
        "--frame",
        choices=("absolute", "relative"),
        default=cfg["frame"],
        help=(
            "'absolute' = values as recorded (default); 'relative' = action "
            "stored as the per-frame increment action - state while the "
            "state itself stays absolute (TCP RPY increments wrapped to "
            "[-pi, pi])"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_paths = [path.expanduser().resolve() for path in args.sessions]
    session_path = session_paths[0]
    suffix = representation_suffix(args.space, args.frame)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output(session_path).resolve().with_name(
            default_output(session_path).resolve().name + suffix
        )
    )
    existing_provenance: dict[str, Any] = {}
    existing_info: dict[str, Any] = {}
    existing_session_metadata: dict[str, Any] = {}
    if args.append:
        if not output.is_dir():
            raise RuntimeError(f"--append requires an existing output dataset: {output}")
        provenance_path = output / "meta" / "piper_conversion.json"
        info_path = output / "meta" / "info.json"
        if not provenance_path.is_file() or not info_path.is_file():
            raise RuntimeError(
                "--append requires a dataset created by convert-piper-raw "
                f"(missing {provenance_path if not provenance_path.is_file() else info_path})"
            )
        existing_provenance = read_json(provenance_path)
        existing_info = read_json(info_path)
        incomplete_marker = output / "CONVERSION_INCOMPLETE.txt"
        if incomplete_marker.exists():
            raise RuntimeError(
                f"target contains {incomplete_marker.name}; repair or rebuild it before appending"
            )
        session_metadata_path = output / "meta" / "piper_raw_session.json"
        if not session_metadata_path.is_file():
            raise RuntimeError(f"existing dataset is missing {session_metadata_path}")
        existing_session_metadata = read_json(session_metadata_path)
        existing_repo_id = existing_provenance.get("repo_id")
        if not existing_repo_id:
            raise RuntimeError("existing conversion metadata has no repo_id")
        if args.repo_id is not None and args.repo_id != existing_repo_id:
            raise RuntimeError(
                f"--repo-id {args.repo_id!r} differs from existing {existing_repo_id!r}"
            )
        repo_id = str(existing_repo_id)
        existing_sessions = {
            str(Path(path).expanduser().resolve())
            for path in existing_provenance.get("source_sessions", [])
        }
        duplicates = [str(path) for path in session_paths if str(path) in existing_sessions]
        if duplicates:
            raise RuntimeError(
                "raw session already exists in the target dataset: " + ", ".join(duplicates)
            )
        representation = existing_provenance.get("representation", {})
        if representation.get("space", "joint") != args.space or representation.get(
            "frame", "absolute"
        ) != args.frame:
            raise RuntimeError(
                "append representation differs from existing dataset: "
                f"existing space={representation.get('space')}, frame={representation.get('frame')}; "
                f"requested space={args.space}, frame={args.frame}"
            )
        existing_roles = list(existing_provenance.get("camera_roles", []))
        if set(existing_roles) != set(camera_roles):
            raise RuntimeError(
                "append camera roles differ from existing dataset: "
                f"existing={existing_roles}, requested={camera_roles}. "
                "Convert into a new dataset instead of appending."
            )
    else:
        repo_id = args.repo_id or default_repo_id(session_path) + suffix
    if args.image_writer_threads < 1 or args.encoder_threads < 1:
        raise RuntimeError("writer/encoder thread counts must be positive")
    if (
        output == session_path
        or session_path in output.parents
        or output in session_path.parents
    ):
        raise RuntimeError(
            "output must be separate from the source session and its parents"
        )

    discovered = [
        discover_episodes(path, args.include_incomplete) for path in session_paths
    ]
    session = discovered[0][0]
    episodes = [episode for _, items, _ in discovered for episode in items]
    skipped = [
        f"{path.name}/{reason}"
        for path, (_, _, reasons) in zip(session_paths, discovered)
        for reason in reasons
    ]
    # Roles the session actually recorded (validated in full below) versus the
    # roles this conversion writes.  Keeping them apart lets ``--camera-role``
    # drop a camera without relaxing source validation: validate_episode still
    # checks every recorded camera's video, matching the raw arrays' per-camera
    # columns, which are sized by the recorded count.
    recorded_roles = list(episodes[0].metadata.get("camera_roles", []))
    if not recorded_roles:
        raise RuntimeError("selected episodes contain no camera roles")
    camera_roles = select_camera_roles(recorded_roles, args.camera_role)

    compatibility_keys = (
        "record_fps", "state_names", "state_units", "action_semantics"
    )
    if args.append:
        for key in compatibility_keys:
            if existing_session_metadata.get(key) != session.get(key):
                raise RuntimeError(
                    f"append session has incompatible {key}: {session.get(key)!r} != "
                    f"{existing_session_metadata.get(key)!r}"
                )
        existing_cameras = existing_session_metadata.get("cameras", {})
        if list(existing_cameras) != list(session.get("cameras", {})):
            raise RuntimeError("append session has incompatible camera roles")
        for role in camera_roles:
            for key in ("serial", "stream"):
                if session["cameras"][role].get(key) != existing_cameras[role].get(key):
                    raise RuntimeError(f"append session has incompatible camera {role} {key}")
    for path, (candidate, _, _) in zip(session_paths[1:], discovered[1:]):
        for key in compatibility_keys:
            if candidate.get(key) != session.get(key):
                raise RuntimeError(
                    f"{path}: incompatible {key}: {candidate.get(key)!r} != "
                    f"{session.get(key)!r}"
                )
        if list(candidate.get("cameras", {})) != list(session.get("cameras", {})):
            raise RuntimeError(f"{path}: incompatible camera roles")
        for role in camera_roles:
            expected = session["cameras"][role]
            actual = candidate["cameras"][role]
            for key in ("serial", "stream"):
                if actual.get(key) != expected.get(key):
                    raise RuntimeError(f"{path}: incompatible camera {role} {key}")

    print("Sources: " + ", ".join(str(path) for path in session_paths))
    print(f"Output: {output}")
    print(f"Repo id: {repo_id}")
    print(f"Representation: space={args.space}, frame={args.frame}")
    camera_note = (
        f"{', '.join(camera_roles)} (of {', '.join(recorded_roles)})"
        if camera_roles != recorded_roles
        else ", ".join(camera_roles)
    )
    print(f"Episodes: {len(episodes)}; cameras: {camera_note}")
    for reason in skipped:
        print(f"Skipping {reason}")

    fk = None
    if args.space == "tcp":
        fk = load_tcp_fk()

    print("Validating source episodes...")
    for episode in episodes:
        validate_episode(episode, episode.session, recorded_roles)
        if fk is not None:
            validate_fk_consistency(
                episode, fk, int(episode.metadata["frames"])
            )
        print(f"  OK {episode.session_path.name}/{episode.path.name}")

    if output.exists() and not args.append:
        if not args.overwrite:
            raise RuntimeError(
                f"output already exists: {output}; use --append to add new "
                "sessions or --overwrite to replace it"
            )
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is unavailable. Run: conda activate robotarm"
        ) from exc

    features = build_features(session, camera_roles, args.space)
    if args.append:
        if int(existing_info.get("fps", -1)) != int(session["record_fps"]):
            raise RuntimeError(
                f"append fps={session['record_fps']} differs from existing fps={existing_info.get('fps')}"
            )
        existing_features = existing_info.get("features", {})
        expected_keys = set(features) | LEROBOT_BOOKKEEPING_KEYS
        if set(existing_features) != expected_keys:
            raise RuntimeError("append feature keys differ from existing dataset")
        for key, expected in features.items():
            actual = existing_features[key]
            if (
                actual.get("dtype") != expected.get("dtype")
                or list(actual.get("shape", [])) != list(expected.get("shape", []))
                or actual.get("names") != expected.get("names")
            ):
                raise RuntimeError(f"append feature {key!r} differs from existing dataset")
    encoder = RGBEncoderConfig(vcodec="h264", crf=23, preset="fast")
    if args.append:
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=output,
            image_writer_threads=args.image_writer_threads,
            rgb_encoder=encoder,
            encoder_threads=args.encoder_threads,
        )
        first_output_index = int(existing_info["total_episodes"])
        existing_frames = int(existing_info["total_frames"])
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            fps=int(session["record_fps"]),
            robot_type="piper_gamepad",
            features=features,
            use_videos=True,
            image_writer_threads=args.image_writer_threads,
            rgb_encoder=encoder,
            encoder_threads=args.encoder_threads,
        )
        first_output_index = 0
        existing_frames = 0

    converted: list[dict[str, Any]] = []
    try:
        for output_index, episode in enumerate(episodes, start=first_output_index):
            print(f"Converting {episode.path.name} -> episode {output_index}...")
            rows = convert_episode(
                dataset, episode, camera_roles, args.space, args.frame, fk
            )
            converted.append(
                {
                    "source_episode": episode.path.name,
                    "source_session": episode.session_path.name,
                    "source_episode_index": episode.metadata.get("episode_index"),
                    "lerobot_episode_index": output_index,
                    "status": episode.metadata.get("status"),
                    "frames": rows,
                    "task": episode.metadata.get("task"),
                }
            )
        dataset.finalize()
    except Exception:
        marker = output / "CONVERSION_INCOMPLETE.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            (
                "Append failed before finalization. Do not append or train from "
                "this dataset until it has been repaired or rebuilt.\n"
                if args.append
                else "Conversion failed before finalization. Remove this output "
                "or rerun with --overwrite.\n"
            ),
            encoding="utf-8",
        )
        raise

    provenance = {
        "converter": "10_convert_piper_raw_to_lerobot.py",
        "compatible_policies": ["act", "diffusion"],
        "converted_at": datetime.now().astimezone().isoformat(),
        "source_format": RAW_FORMAT,
        "source_sessions": [
            *existing_provenance.get("source_sessions", []),
            *(str(path) for path in session_paths),
        ],
        "repo_id": repo_id,
        "include_incomplete": bool(
            existing_provenance.get("include_incomplete", False) or args.include_incomplete
        ),
        "camera_roles": camera_roles,
        # Only present when --camera-role dropped a recorded camera, so the
        # audit trail shows what the source had.
        **(
            {"available_camera_roles": recorded_roles}
            if camera_roles != recorded_roles
            else {}
        ),
        "representation": {
            "space": args.space,
            "frame": args.frame,
            "state_semantics": (
                "end-effector pose [x,y,z,r,p,y] + gripper width (m), MDH FK "
                "consistent with recording-time robot.fk"
                if args.space == "tcp"
                else "6 joint angles (rad) + gripper width (m)"
            ),
            "action_semantics": (
                "per-frame increment action - state (state itself stays "
                "absolute); TCP RPY increments wrapped to [-pi, pi]"
                if args.frame == "relative"
                else (
                    "end-effector pose target + gripper target (absolute)"
                    if args.space == "tcp"
                    else "commanded MOVE_JS joint targets and gripper target"
                )
            ),
        },
        "model_observation_keys": list(MODEL_OBSERVATION_KEYS),
        "training_target_key": "action",
        "excluded_noncausal_fields": [
            "observation.teleop_delta",
            "observation.raw_timestamp",
            "observation.camera_skew",
            "observation.camera.*.host_timestamp",
            "observation.camera.*.sensor_timestamp",
            "observation.camera.*.frame_number",
            "observation.camera.*.frame_age",
        ],
        "excluded_redundant_fields": ["observation.tcp_pose"],
        "episodes": [*existing_provenance.get("episodes", []), *converted],
    }
    existing_session_records = []
    existing_session_metadata_path = output / "meta" / "piper_raw_session.json"
    if args.append and existing_session_metadata_path.is_file():
        existing_session_records = read_json(existing_session_metadata_path).get(
            "merged_source_sessions", []
        )
    new_session_records = [
        {
            "path": str(path),
            "created_at": item[0].get("created_at"),
            "setup_id": item[0].get("setup_id"),
            "setup_note": item[0].get("setup_note"),
            "episodes": len(item[1]),
        }
        for path, item in zip(session_paths, discovered)
    ]
    merged_session = dict(existing_session_metadata if args.append else session)
    merged_session["merged_source_sessions"] = [
        *existing_session_records,
        *new_session_records,
    ]
    write_json(output / "meta" / "piper_raw_session.json", merged_session)
    write_json(output / "meta" / "piper_conversion.json", provenance)

    validate_act_output_schema(output, camera_roles)

    loaded = LeRobotDataset(repo_id=repo_id, root=output, download_videos=False)
    expected_frames = existing_frames + sum(item["frames"] for item in converted)
    if len(loaded) != expected_frames:
        raise RuntimeError(
            f"read-back length is {len(loaded)}, expected {expected_frames}"
        )
    expected_episodes = first_output_index + len(converted)
    if int(loaded.meta.total_episodes) != expected_episodes:
        raise RuntimeError(
            f"read-back episode count is {loaded.meta.total_episodes}, expected {expected_episodes}"
        )
    incomplete_marker = output / "CONVERSION_INCOMPLETE.txt"
    if incomplete_marker.exists():
        incomplete_marker.unlink()
    print(
        f"Done: {'appended' if args.append else 'converted'} {len(converted)} episodes; "
        f"dataset now has {expected_episodes} episodes, "
        f"{expected_frames} frames -> {output}"
    )
    state_desc = (
        "tcp_pose+gripper" if args.space == "tcp" else "joints+gripper"
    )
    action_desc = (
        "delta(action-state)" if args.frame == "relative" else "absolute"
    )
    print(
        f"Schema: observation.state[{state_desc}] + "
        + " + ".join(f"observation.images.{role}" for role in camera_roles)
        + f" -> action[{action_desc}]"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Conversion failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
