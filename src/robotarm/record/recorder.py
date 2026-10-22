"""与遥操作输入源无关的原始 episode 记录器。

RealSense RGB + MP4（ffmpeg 异步编码）+ NPZ 样本数组 + JSON 元数据，
不依赖 LeRobot。手柄与 XR/PICO 两条采集链路共用本模块。

迁移自 piper_ctrl/control_scripts/src/raw_dataset_recorder.py。
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.camera.realsense import RoleCameras
from .schema import (
    FORMAT_VERSION,
    SAMPLE_FIELDS,
    session_metadata,
    sample_dtype,
)


def _json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class FFmpegWriter:
    """Bounded asynchronous raw-BGR to H.264 writer."""

    _END = object()

    def __init__(
        self, path: Path, width: int, height: int, fps: int, queue_size: int = 60
    ) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is not installed; run: sudo apt install ffmpeg")
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                image = self.queue.get()
                if image is self._END:
                    break
                if self.process.stdin is None:
                    raise RuntimeError("ffmpeg stdin is unavailable")
                self.process.stdin.write(image.tobytes())
        except BaseException as exc:
            self.error = exc

    def submit(self, image: Any) -> None:
        if self.error is not None:
            raise RuntimeError(f"ffmpeg writer failed: {self.error}") from self.error
        try:
            self.queue.put_nowait(image)
        except queue.Full as exc:
            raise RuntimeError("video encoder queue is full; episode is incomplete") from exc

    def close(self) -> None:
        if self.error is not None:
            self.process.kill()
            self._thread.join(timeout=2.0)
            raise RuntimeError(f"ffmpeg writer failed: {self.error}") from self.error
        try:
            self.queue.put(self._END, timeout=5.0)
        except queue.Full as exc:
            self.process.kill()
            raise RuntimeError("ffmpeg queue did not drain during close") from exc
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            self.process.kill()
            raise RuntimeError("ffmpeg writer did not finish within 30 seconds")
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = b"" if self.process.stderr is None else self.process.stderr.read()
        code = self.process.wait(timeout=10.0)
        if self.error is not None:
            raise RuntimeError(f"ffmpeg writer failed: {self.error}") from self.error
        if code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {code}: {stderr.decode(errors='replace').strip()}"
            )


class RawEpisodeRecorder:
    """Record synchronized policy samples without depending on LeRobot."""

    def __init__(
        self,
        root_base: Path,
        task: str,
        setup_id: str,
        setup_note: str,
        fps: int,
        width: int,
        height: int,
        camera_serials: dict[str, str],
        robot_metadata: dict[str, Any],
        preview: bool = True,
        encoder_queue_size: int = 60,
        extra_sample_fields: tuple[str, ...] = (),
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required; install it with: sudo apt install ffmpeg")
        free = shutil.disk_usage(root_base.resolve().parent if root_base.exists() else Path.cwd()).free
        if free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free; refusing to start data collection")
        if free < 10 * 1024**3:
            print(f"[record] WARNING: only {free / 1024**3:.1f} GiB free.")
        self.fps = fps
        self.width = width
        self.height = height
        self.task = task
        self.encoder_queue_size = encoder_queue_size
        overlapping_fields = set(extra_sample_fields) & set(SAMPLE_FIELDS)
        if overlapping_fields:
            raise ValueError(
                "extra sample fields overlap built-in fields: "
                f"{sorted(overlapping_fields)}"
            )
        if len(set(extra_sample_fields)) != len(extra_sample_fields):
            raise ValueError("extra sample fields must be unique")
        self.extra_sample_fields = tuple(extra_sample_fields)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.root = root_base / stamp
        self.root.mkdir(parents=True, exist_ok=False)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir()
        try:
            self.cameras = RoleCameras(
                camera_serials, width, height, fps, preview, always_on_top=True
            )
            self.cameras.wait_ready()
        except Exception:
            cameras = getattr(self, "cameras", None)
            if cameras is not None:
                cameras.close()
            shutil.rmtree(self.root, ignore_errors=True)
            raise
        self.session = session_metadata(
            task, setup_id, setup_note, fps, robot_metadata, self.cameras.metadata
        )
        self.session["raw_sample_fields"] = [
            *SAMPLE_FIELDS, *self.extra_sample_fields
        ]
        _json_write(self.root / "session.json", self.session)
        self.active = False
        self.episode_index = 0
        self.frames = 0
        self.next_frame_time = 0.0
        self.started_at = 0.0
        self.working_dir: Path | None = None
        self.writers: dict[str, FFmpegWriter] = {}
        self.samples: dict[str, list[Any]] = {}
        self.dropped_samples = 0
        print(f"[record] Raw dataset session: {self.root.resolve()}")

    def preview(self) -> None:
        frames = self.cameras.latest()
        self.cameras.show(frames, self.active)

    def start_episode(self) -> None:
        if self.active:
            return
        free = shutil.disk_usage(self.root).free
        if free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free; cannot start another episode")
        self.working_dir = self.episodes_dir / f".episode_{self.episode_index:06d}.in_progress"
        self.working_dir.mkdir()
        self.writers = {}
        try:
            for role in self.cameras.workers:
                self.writers[role] = FFmpegWriter(
                    self.working_dir / f"{role}.mp4", self.width, self.height,
                    self.fps, self.encoder_queue_size,
                )
        except Exception:
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            shutil.rmtree(self.working_dir)
            self.working_dir = None
            raise
        self.samples = {
            field: [] for field in (*SAMPLE_FIELDS, *self.extra_sample_fields)
        }
        self.frames = 0
        self.dropped_samples = 0
        self.started_at = time.monotonic()
        self.next_frame_time = self.started_at
        self.active = True
        print(f"\n[record] Episode {self.episode_index}; short save, long discard.")

    def capture(
        self,
        now: float,
        state: list[float],
        action: list[float],
        tcp_pose: list[float],
        teleop_delta: list[float],
        extra_samples: dict[str, Any] | None = None,
        *, show_preview: bool = True,
    ) -> bool:
        frames = self.cameras.latest()
        if show_preview:
            self.cameras.show(frames, self.active)
        if not self.active or now < self.next_frame_time:
            return False
        extra_samples = extra_samples or {}
        expected_extra = set(self.extra_sample_fields)
        if set(extra_samples) != expected_extra:
            raise RuntimeError(
                "extra sample keys differ from recorder schema: "
                f"expected={sorted(expected_extra)}, got={sorted(extra_samples)}"
            )
        host_times = [sample[1] for sample in frames.values()]
        ages = [now - value for value in host_times]
        if max(ages) > 0.25:
            self.dropped_samples += 1
            self.next_frame_time = now + 1.0 / self.fps
            if max(ages) > 1.0:
                raise RuntimeError(f"camera frame is stale ({max(ages) * 1000:.0f} ms)")
            return False
        for role, (image, _, _, _) in frames.items():
            self.writers[role].submit(image)
        roles = list(frames)
        self.samples["timestamp"].append(now - self.started_at)
        self.samples["observation_state"].append(state)
        self.samples["action"].append(action)
        self.samples["observation_tcp_pose"].append(tcp_pose)
        self.samples["teleop_delta"].append(teleop_delta)
        for field in self.extra_sample_fields:
            self.samples[field].append(extra_samples[field])
        self.samples["camera_host_timestamp"].append(
            [frames[role][1] - self.started_at for role in roles]
        )
        self.samples["camera_sensor_timestamp"].append(
            [frames[role][2] for role in roles]
        )
        self.samples["camera_frame_number"].append(
            [frames[role][3] for role in roles]
        )
        self.samples["camera_frame_age"].append(ages)
        self.samples["camera_skew"].append(max(host_times) - min(host_times))
        self.frames += 1
        self.next_frame_time += 1.0 / self.fps
        if self.next_frame_time < now - 1.0 / self.fps:
            self.dropped_samples += 1
            self.next_frame_time = now + 1.0 / self.fps
        return True

    def _finish(self, status: str) -> Path | None:
        if not self.active or self.working_dir is None:
            return None
        self.active = False
        writer_errors = []
        for role, writer in self.writers.items():
            try:
                writer.close()
            except Exception as exc:
                writer_errors.append(f"{role}: {exc}")
        if writer_errors and status != "discarded":
            status = "incomplete"
        import numpy as np

        arrays = {
            key: np.asarray(value, dtype=sample_dtype(key))
            for key, value in self.samples.items()
        }
        np.savez_compressed(self.working_dir / "frames.npz", **arrays)
        duration = 0.0 if not self.samples["timestamp"] else self.samples["timestamp"][-1]
        skew = self.samples["camera_skew"]
        metadata = {
            "format_version": FORMAT_VERSION,
            "episode_index": self.episode_index,
            "task": self.task,
            "status": status,
            "frames": self.frames,
            "duration_seconds": duration,
            "camera_roles": list(self.cameras.workers),
            "dropped_samples": self.dropped_samples,
            "maximum_camera_skew_seconds": max(skew) if skew else 0.0,
            "writer_errors": writer_errors,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        _json_write(self.working_dir / "episode.json", metadata)
        if status == "discarded":
            shutil.rmtree(self.working_dir)
            print(f"\n[record] Episode {self.episode_index} discarded.")
            result = None
        else:
            prefix = "episode" if status == "complete" and self.frames else "incomplete_episode"
            destination = self.episodes_dir / f"{prefix}_{self.episode_index:06d}"
            self.working_dir.replace(destination)
            print(
                f"\n[record] Episode {self.episode_index} {status}: "
                f"{self.frames} frames -> {destination}"
            )
            result = destination
        self.episode_index += 1
        self.working_dir = None
        self.writers = {}
        self.samples = {}
        return result

    def save_episode(self) -> Path | None:
        return self._finish("complete" if self.frames else "incomplete")

    def discard_episode(self) -> None:
        self._finish("discarded")

    def mark_incomplete(self) -> Path | None:
        return self._finish("incomplete")

    def close(self) -> None:
        if self.active:
            self.mark_incomplete()
        self.cameras.close()
