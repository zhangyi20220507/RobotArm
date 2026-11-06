"""Non-blocking multi-format recordings for the online policy runner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any

from ..record.recorder import FFmpegWriter


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class _ActiveRecording:
    index: int
    working_dir: Path
    started_monotonic: float
    started_wall_time: float
    next_frame_time: float
    writers: dict[str, Any]
    samples: dict[str, list[Any]] = field(default_factory=dict)
    dropped_samples: int = 0
    failure: str | None = None


class PolicyRecordingManager:
    """Record policy evaluation without owning cameras or blocking control."""

    def __init__(
        self,
        root_base: Path,
        fps: int,
        width: int,
        height: int,
        camera_roles: list[str],
        camera_metadata: dict[str, Any],
        session_metadata: dict[str, Any],
        encoder_queue_size: int = 120,
        writer_factory=None,
        cv2_module=None,
        np_module=None,
    ) -> None:
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("recording dimensions and fps must be positive")
        if encoder_queue_size < 1:
            raise ValueError("recording encoder queue size must be positive")
        if not camera_roles or len(set(camera_roles)) != len(camera_roles):
            raise ValueError("recording camera roles must be non-empty and unique")
        if set(camera_roles) != set(camera_metadata):
            raise ValueError("recording camera metadata does not match camera roles")
        if writer_factory is None:
            writer_factory = FFmpegWriter
        if cv2_module is None:
            import cv2 as cv2_module
        if np_module is None:
            import numpy as np_module

        self.root_base = Path(root_base)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.roles = list(camera_roles)
        self.camera_metadata = camera_metadata
        self.session_metadata = dict(session_metadata)
        self.encoder_queue_size = int(encoder_queue_size)
        self.writer_factory = writer_factory
        self.cv2 = cv2_module
        self.np = np_module
        self.session_root: Path | None = None
        self.active_recording: _ActiveRecording | None = None
        self.next_index = 0
        self._finishers: list[threading.Thread] = []
        self._messages: queue.SimpleQueue[str] = queue.SimpleQueue()

    @property
    def active(self) -> bool:
        return self.active_recording is not None

    def _create_session(self) -> bool:
        if self.session_root is not None:
            return False
        usage_path = self.root_base
        while not usage_path.exists() and usage_path != usage_path.parent:
            usage_path = usage_path.parent
        free = shutil.disk_usage(usage_path).free
        if free < 2 * 1024**3:
            raise RuntimeError("less than 2 GiB free; policy recording refused")
        if free < 10 * 1024**3:
            self._messages.put(
                f"[record] WARNING: only {free / 1024**3:.1f} GiB free"
            )
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.root_base.mkdir(parents=True, exist_ok=True)
        root = self.root_base / stamp
        root.mkdir(exist_ok=False)
        metadata = {
            "format": "robotarm_online_policy_recording_v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "record_fps": self.fps,
            "frame_size": [self.width, self.height],
            "camera_roles": self.roles,
            "cameras": self.camera_metadata,
            **self.session_metadata,
        }
        _write_json(root / "session.json", metadata)
        self.session_root = root
        return True

    def start(self) -> Path:
        if self.active:
            assert self.active_recording is not None
            return self.active_recording.working_dir
        created_session = self._create_session()
        assert self.session_root is not None
        working = self.session_root / f".recording_{self.next_index:06d}.in_progress"
        working.mkdir()
        writers: dict[str, Any] = {}
        try:
            for role in self.roles:
                writers[role] = self.writer_factory(
                    working / f"{role}.mp4",
                    self.width,
                    self.height,
                    self.fps,
                    self.encoder_queue_size,
                )
            writers["combined"] = self.writer_factory(
                working / "combined.mp4",
                self.width * len(self.roles),
                self.height,
                self.fps,
                self.encoder_queue_size,
            )
        except Exception:
            for writer in writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            shutil.rmtree(working, ignore_errors=True)
            if created_session:
                shutil.rmtree(self.session_root, ignore_errors=True)
                self.session_root = None
            raise
        now = time.monotonic()
        sample_names = (
            "timestamp", "wall_time", "observation_state", "action",
            "raw_action", "camera_host_timestamp", "camera_sensor_timestamp",
            "camera_frame_number", "camera_frame_age", "camera_skew",
            "trial_id", "motion_enabled", "command_sent", "control_event",
        )
        self.active_recording = _ActiveRecording(
            index=self.next_index,
            working_dir=working,
            started_monotonic=now,
            started_wall_time=time.time(),
            next_frame_time=now,
            writers=writers,
            samples={name: [] for name in sample_names},
        )
        self.next_index += 1
        self._messages.put(
            f"[record] Recording {self.active_recording.index} started; "
            "press V again to save"
        )
        return working

    def _combined_frame(
        self,
        recording: _ActiveRecording,
        frames: dict[str, tuple[Any, float, float, int]],
        trial_id: int,
        motion_enabled: bool,
    ) -> Any:
        views = []
        elapsed = max(0.0, time.monotonic() - recording.started_monotonic)
        for role in self.roles:
            view = frames[role][0].copy()
            self.cv2.putText(
                view,
                role,
                (12, 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            views.append(view)
        combined = self.np.hstack(views)
        status = "EXEC" if motion_enabled else "PAUSED/HOMING"
        self.cv2.putText(
            combined,
            f"REC {elapsed:7.2f}s  trial={trial_id}  {status}",
            (12, self.height - 16),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        return combined

    def capture(
        self,
        now: float,
        frames: dict[str, tuple[Any, float, float, int]],
        state: Any,
        action: Any,
        raw_action: Any,
        trial_id: int,
        motion_enabled: bool,
        command_sent: bool,
        control_event: str | None = None,
    ) -> bool:
        recording = self.active_recording
        if recording is None or now < recording.next_frame_time:
            return False
        try:
            if set(frames) != set(self.roles):
                raise RuntimeError("camera roles changed during policy recording")
            for role in self.roles:
                if getattr(frames[role][0], "shape", None) != (
                    self.height, self.width, 3
                ):
                    raise RuntimeError(
                        f"{role} recording frame has unexpected shape "
                        f"{getattr(frames[role][0], 'shape', None)}"
                    )
            state = self.np.asarray(state, dtype=self.np.float32)
            action = self.np.asarray(action, dtype=self.np.float32)
            raw_action = self.np.asarray(raw_action, dtype=self.np.float32)
            if any(value.shape != (7,) for value in (state, action, raw_action)):
                raise RuntimeError("policy recording requires 7D state/actions")
            if not self.np.isfinite(state).all() or not self.np.isfinite(action).all():
                raise RuntimeError("policy recording received non-finite state/safe action")
            host_times = [frames[role][1] for role in self.roles]
            ages = [max(0.0, now - value) for value in host_times]
            if max(ages) > 1.0:
                raise RuntimeError(
                    f"camera frame is stale ({max(ages) * 1000:.0f} ms)"
                )
            if max(ages) > 0.25:
                recording.dropped_samples += 1
                recording.next_frame_time = now + 1.0 / self.fps
                return False
            combined = self._combined_frame(
                recording, frames, int(trial_id), bool(motion_enabled)
            )
            for role in self.roles:
                recording.writers[role].submit(frames[role][0])
            recording.writers["combined"].submit(combined)
            samples = recording.samples
            samples["timestamp"].append(now - recording.started_monotonic)
            samples["wall_time"].append(time.time())
            samples["observation_state"].append(state.copy())
            samples["action"].append(action.copy())
            samples["raw_action"].append(raw_action.copy())
            samples["camera_host_timestamp"].append(
                [frames[role][1] - recording.started_monotonic for role in self.roles]
            )
            samples["camera_sensor_timestamp"].append(
                [frames[role][2] for role in self.roles]
            )
            samples["camera_frame_number"].append(
                [frames[role][3] for role in self.roles]
            )
            samples["camera_frame_age"].append(ages)
            samples["camera_skew"].append(max(host_times) - min(host_times))
            samples["trial_id"].append(int(trial_id))
            samples["motion_enabled"].append(bool(motion_enabled))
            samples["command_sent"].append(bool(command_sent))
            samples["control_event"].append(control_event or "")
            recording.next_frame_time += 1.0 / self.fps
            if recording.next_frame_time < now - 1.0 / self.fps:
                recording.dropped_samples += 1
                recording.next_frame_time = now + 1.0 / self.fps
            return True
        except Exception as exc:
            self._messages.put(
                f"[record] Recording {recording.index} failed: "
                f"{type(exc).__name__}: {exc}; model control continues"
            )
            self._detach_and_finish("incomplete", str(exc))
            return False

    def _arrays(self, recording: _ActiveRecording) -> dict[str, Any]:
        np = self.np
        count = len(recording.samples["timestamp"])
        roles = len(self.roles)
        arrays = {
            "timestamp": np.asarray(recording.samples["timestamp"], dtype=np.float64),
            "wall_time": np.asarray(recording.samples["wall_time"], dtype=np.float64),
            "observation_state": np.asarray(
                recording.samples["observation_state"], dtype=np.float32
            ).reshape(count, 7),
            "action": np.asarray(recording.samples["action"], dtype=np.float32).reshape(count, 7),
            "raw_action": np.asarray(
                recording.samples["raw_action"], dtype=np.float32
            ).reshape(count, 7),
            "camera_host_timestamp": np.asarray(
                recording.samples["camera_host_timestamp"], dtype=np.float64
            ).reshape(count, roles),
            "camera_sensor_timestamp": np.asarray(
                recording.samples["camera_sensor_timestamp"], dtype=np.float64
            ).reshape(count, roles),
            "camera_frame_number": np.asarray(
                recording.samples["camera_frame_number"], dtype=np.int64
            ).reshape(count, roles),
            "camera_frame_age": np.asarray(
                recording.samples["camera_frame_age"], dtype=np.float64
            ).reshape(count, roles),
            "camera_skew": np.asarray(recording.samples["camera_skew"], dtype=np.float64),
            "trial_id": np.asarray(recording.samples["trial_id"], dtype=np.int64),
            "motion_enabled": np.asarray(
                recording.samples["motion_enabled"], dtype=np.bool_
            ),
            "command_sent": np.asarray(recording.samples["command_sent"], dtype=np.bool_),
            "control_event": np.asarray(recording.samples["control_event"], dtype="U32"),
        }
        return arrays

    def _write_joint_csv(self, path: Path, arrays: dict[str, Any]) -> None:
        header = ["timestamp_s", "wall_time_s", "trial_id", "motion_enabled", "command_sent"]
        header += [f"joint_{index}_rad" for index in range(1, 7)] + ["gripper_width_m"]
        header += [f"joint_{index}_deg" for index in range(1, 7)] + ["gripper_width_mm"]
        header += [f"action_joint_{index}_rad" for index in range(1, 7)] + ["action_gripper_m"]
        header += [f"raw_joint_{index}_rad" for index in range(1, 7)] + ["raw_gripper_m"]
        header += ["control_event"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index, state in enumerate(arrays["observation_state"]):
                writer.writerow([
                    float(arrays["timestamp"][index]),
                    float(arrays["wall_time"][index]),
                    int(arrays["trial_id"][index]),
                    int(arrays["motion_enabled"][index]),
                    int(arrays["command_sent"][index]),
                    *[float(value) for value in state],
                    *[math.degrees(float(value)) for value in state[:6]],
                    float(state[6]) * 1000.0,
                    *[float(value) for value in arrays["action"][index]],
                    *[float(value) for value in arrays["raw_action"][index]],
                    str(arrays["control_event"][index]),
                ])

    def _finish(
        self, recording: _ActiveRecording, status: str, failure: str | None
    ) -> None:
        writer_errors = []
        for name, writer in recording.writers.items():
            try:
                writer.close()
            except Exception as exc:
                writer_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if writer_errors:
            status = "incomplete"
        try:
            arrays = self._arrays(recording)
            self.np.savez_compressed(recording.working_dir / "frames.npz", **arrays)
            self._write_joint_csv(recording.working_dir / "joints.csv", arrays)
            count = len(arrays["timestamp"])
            metadata = {
                "format": "robotarm_online_policy_recording_v1",
                "recording_index": recording.index,
                "status": status,
                "frames": count,
                "duration_seconds": (
                    0.0 if count == 0 else float(arrays["timestamp"][-1])
                ),
                "camera_roles": self.roles,
                "videos": {
                    "combined": "combined.mp4",
                    "separate": {role: f"{role}.mp4" for role in self.roles},
                },
                "state": "six joint angles in radians plus gripper width in metres",
                "action": "safe command target when sent; measured state while paused",
                "raw_action": "unshaped policy output",
                "dropped_samples": recording.dropped_samples,
                "failure": failure,
                "writer_errors": writer_errors,
                "started_at": datetime.fromtimestamp(
                    recording.started_wall_time
                ).astimezone().isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
            }
            _write_json(recording.working_dir / "recording.json", metadata)
            assert self.session_root is not None
            prefix = "recording" if status == "complete" and count else "incomplete_recording"
            destination = self.session_root / f"{prefix}_{recording.index:06d}"
            recording.working_dir.replace(destination)
            self._messages.put(
                f"[record] Recording {recording.index} {status}: "
                f"{count} frames -> {destination}"
            )
        except Exception as exc:
            self._messages.put(
                f"[record] Could not finalize recording {recording.index}: "
                f"{type(exc).__name__}: {exc}; files remain in {recording.working_dir}"
            )

    def _detach_and_finish(self, status: str, failure: str | None = None) -> Path | None:
        recording = self.active_recording
        if recording is None:
            return None
        self.active_recording = None
        thread = threading.Thread(
            target=self._finish,
            args=(recording, status, failure),
            name=f"policy-recording-finalize-{recording.index}",
            daemon=True,
        )
        self._finishers.append(thread)
        thread.start()
        return recording.working_dir

    def stop(self) -> Path | None:
        if not self.active:
            return None
        self._messages.put("[record] Recording stopped; finalizing videos in background")
        return self._detach_and_finish("complete")

    def drain_messages(self) -> list[str]:
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                return messages

    def close(self) -> None:
        if self.active:
            self._detach_and_finish(
                "incomplete", "program exited before recording was stopped with V"
            )
        for thread in self._finishers:
            thread.join(timeout=40.0)
            if thread.is_alive():
                self._messages.put(
                    f"[record] WARNING: {thread.name} did not finish within 40s"
                )

