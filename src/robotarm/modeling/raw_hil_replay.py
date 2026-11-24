"""Read reviewed full HG trials directly into an RL replay buffer.

This avoids the RAW -> LeRobot MP4 transcode. The reader is single-threaded;
LeRobot's ReplayBuffer.from_lerobot_dataset iterates it in index order.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

from ..record import convert
from ..record.convert_hil_serl import selected_trials


class RawHILReplayDataset:
    def __init__(self, roots: list[Path] | Path):
        if isinstance(roots, (str, Path)):
            roots = [Path(roots)]
        self.trials: list[dict[str, Any]] = []
        self.offsets: list[int] = []
        self.length = 0
        self.camera_roles: list[str] | None = None
        for root in roots:
            root = Path(root).expanduser().resolve()
            session, episodes, _ = convert.discover_episodes(root, False)
            if not convert.uses_expert_mask(session):
                raise ValueError(f"{root}: RAW HIL replay requires full HG trials")
            selected, _ = selected_trials(root, episodes)
            for item in selected:
                episode = item["episode"]
                roles = list(episode.metadata["camera_roles"])
                if self.camera_roles is None:
                    self.camera_roles = roles
                elif roles != self.camera_roles:
                    raise ValueError("all RAW HIL trials must have the same camera roles")
                self.offsets.append(self.length)
                self.trials.append(item)
                self.length += int(episode.metadata["frames"])
        if not self.trials:
            raise ValueError("no reviewed full HG trials")
        self._active_trial = -1
        self._arrays: dict[str, Any] = {}
        self._videos: dict[str, Any] = {}
        self._video_next_frame = 0
        self._cached_video_frame = -1
        self._cached_images: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        for video in self._videos.values():
            video.release()
        self._videos = {}
        self._arrays = {}
        self._active_trial = -1
        self._cached_images = {}

    def _activate(self, trial_index: int) -> None:
        import cv2
        import numpy as np

        if trial_index == self._active_trial:
            return
        self.close()
        episode = self.trials[trial_index]["episode"]
        with np.load(episode.path / "frames.npz", allow_pickle=False) as data:
            self._arrays = {key: data[key] for key in (
                "observation_state", "action", "control_source", "expert_mask",
                "intervention_id", "command_sent", "novice_action",
            )}
        rows = int(episode.metadata["frames"])
        if any(value.shape[0] != rows for value in self._arrays.values()):
            raise ValueError(f"{episode.path}: NPZ frame count mismatch")
        if self._arrays["observation_state"].shape != (rows, 7) or self._arrays["action"].shape != (rows, 7):
            raise ValueError(f"{episode.path}: expected 7D Piper state/action")
        self._videos = {
            role: cv2.VideoCapture(str(episode.path / f"{role}.mp4"))
            for role in self.camera_roles or []
        }
        if not all(video.isOpened() for video in self._videos.values()):
            self.close()
            raise ValueError(f"{episode.path}: cannot open camera video")
        self._video_next_frame = 0
        self._cached_video_frame = -1
        self._active_trial = trial_index

    def _images(self, frame: int) -> dict[str, Any]:
        import cv2
        import torch

        if frame == self._cached_video_frame:
            return self._cached_images
        if frame != self._video_next_frame:
            for video in self._videos.values():
                if not video.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise RuntimeError(f"could not seek RAW video to frame {frame}")
            self._video_next_frame = frame
        images = {}
        for role, video in self._videos.items():
            ok, bgr = video.read()
            if not ok or bgr is None:
                raise RuntimeError(f"RAW {role} video ended at frame {frame}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images[f"observation.images.{role}"] = (
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            )
        self._video_next_frame = frame + 1
        self._cached_video_frame = frame
        self._cached_images = images
        return images

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        if not 0 <= index < self.length:
            raise IndexError(index)
        trial_index = bisect_right(self.offsets, index) - 1
        self._activate(trial_index)
        frame = index - self.offsets[trial_index]
        arrays = self._arrays
        item = self.trials[trial_index]
        rows = int(item["episode"].metadata["frames"])
        done = frame == rows - 1
        source = int(arrays["control_source"][frame])
        novice = np.asarray(arrays["novice_action"][frame], dtype=np.float32)
        novice_valid = source == 0 and bool(arrays["command_sent"][frame]) and np.isfinite(novice).all()
        return {
            "observation.state": torch.from_numpy(np.asarray(arrays["observation_state"][frame], dtype=np.float32).copy()),
            "action": torch.from_numpy(np.asarray(arrays["action"][frame], dtype=np.float32).copy()),
            "episode_index": torch.tensor(trial_index, dtype=torch.int64),
            "frame_index": torch.tensor(frame, dtype=torch.int64),
            "next.reward": torch.tensor([float(done and item["outcome"] == "success")], dtype=torch.float32),
            "next.done": torch.tensor([done], dtype=torch.bool),
            "complementary_info.is_intervention": torch.tensor([bool(arrays["expert_mask"][frame])]),
            "complementary_info.control_source": torch.tensor([source], dtype=torch.int64),
            "complementary_info.intervention_id": torch.tensor([int(arrays["intervention_id"][frame])], dtype=torch.int64),
            "complementary_info.command_sent": torch.tensor([bool(arrays["command_sent"][frame])]),
            "complementary_info.novice_action": torch.from_numpy(novice.copy() if novice_valid else np.zeros(7, dtype=np.float32)),
            "complementary_info.novice_action_valid": torch.tensor([novice_valid]),
            **self._images(frame),
        }
