"""Read reviewed BC LeRobot data as RL demonstrations without re-encoding video.

The original dataset stays untouched. Sparse terminal rewards are derived from
manual RAW reviews embedded in ``meta/piper_conversion.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def reviewed_episodes(root: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "meta" / "piper_conversion.json").read_text())
    if manifest.get("includes_expert_mask_exports"):
        raise ValueError("offline demonstrations require full-human BC episodes, not HG expert spans")
    selected: dict[int, dict[str, Any]] = {}
    for episode in manifest.get("episodes", []):
        review = episode.get("review") or {}
        if review.get("status") != "keep" or review.get("review_origin") != "manual":
            continue
        outcome = (review.get("labels") or {}).get("task_outcome")
        if outcome not in {"success", "failure"}:
            continue
        index = int(episode["lerobot_episode_index"])
        frames = int(episode["frames"])
        if index in selected or frames < 1:
            raise ValueError(f"invalid or duplicate reviewed episode {index}")
        selected[index] = {"frames": frames, "outcome": outcome}
    if not selected:
        raise ValueError("no manually kept BC episodes with success/failure labels")
    return str(manifest["repo_id"]), selected


class ReviewedOfflineRLDataset:
    """Expose sparse reward/done fields to LeRobot's ReplayBuffer loader."""

    def __init__(self, dataset: Any, episodes: dict[int, dict[str, Any]]):
        self.dataset = dataset
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        sample = dict(self.dataset[index])
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        if episode not in self.episodes:
            raise ValueError(f"episode {episode} is not in the reviewed BC selection")
        selected = self.episodes[episode]
        if not 0 <= frame < selected["frames"]:
            raise ValueError(f"episode {episode} has out-of-range frame {frame}")
        done = frame == selected["frames"] - 1
        sample["next.done"] = torch.tensor([done], dtype=torch.bool)
        sample["next.reward"] = torch.tensor(
            [float(done and selected["outcome"] == "success")], dtype=torch.float32
        )
        return sample


def open_reviewed_offline_rl(root: Path) -> ReviewedOfflineRLDataset:
    """Open only labelled BC episodes; existing MP4/Parquet files are reused."""
    from lerobot.datasets import LeRobotDataset

    root = Path(root).expanduser().resolve()
    repo_id, episodes = reviewed_episodes(root)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=sorted(episodes),
        download_videos=False,
    )
    return ReviewedOfflineRLDataset(dataset, episodes)
