"""Bind an operator's success/failure judgment to a completed RAW trial."""

from __future__ import annotations

from pathlib import Path


def save_manual_outcome(session_root: Path, episode_path: Path, outcome: str) -> int:
    if outcome not in {"success", "failure"}:
        raise ValueError("outcome must be success or failure")
    from .raw_dataset_browser import RawDatasetStore

    store = RawDatasetStore(session_root)
    episode_path = Path(episode_path).resolve()
    for index, source in enumerate(store.sources):
        if source.path.resolve() == episode_path:
            existing = store.reviews().get(str(index), {})
            if existing.get("status") and existing.get("review_origin") == "manual":
                raise ValueError(f"{episode_path.name} already has a manual review")
            store.save_review(
                index, "keep", "Operator outcome at end of HIL trial",
                labels={"task_outcome": outcome},
            )
            return index
    raise FileNotFoundError(f"completed episode not found in {session_root}: {episode_path}")
