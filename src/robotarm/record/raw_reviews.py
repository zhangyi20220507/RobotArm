"""Non-destructive raw episode reviews, bound to source identities and NPZ data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT = "piper_raw_reviews_v1"


def default_raw_reviews_path(root: Path) -> Path:
    return root / "annotations" / "reviews.json"


def episode_identity(episode) -> str:
    origin = episode.metadata.get("merge_source", {})
    source = origin.get("episode_path", str(episode.path))
    digest = hashlib.sha256()
    digest.update(str(Path(source).resolve()).encode())
    digest.update(str(episode.metadata["frames"]).encode())
    with (episode.path / "frames.npz").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_document(root: Path, episodes: list, reviews: dict) -> dict:
    return {
        "format_version": FORMAT,
        "dataset_root": str(root.resolve()),
        "episode_ids": [episode_identity(episode) for episode in episodes],
        "episodes": reviews,
    }


def load_raw_reviews(root: Path, episodes: list, path: Path | None = None,
                     document: dict | None = None, *, allow_append: bool = False) -> dict:
    explicit_path = path is not None
    path = path.expanduser() if path is not None else default_raw_reviews_path(root)
    if document is None:
        if not path.exists():
            if explicit_path:
                raise FileNotFoundError(path)
            return {}
        document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format_version") != FORMAT:
        raise ValueError("invalid raw review format")
    if Path(document.get("dataset_root", "")).resolve() != root.resolve():
        raise ValueError("raw reviews belong to another dataset")
    current_ids = [episode_identity(episode) for episode in episodes]
    stored_ids = document.get("episode_ids")
    identities_match = (
        stored_ids == current_ids
        or (
            allow_append and isinstance(stored_ids, list)
            and len(stored_ids) <= len(current_ids)
            and stored_ids == current_ids[:len(stored_ids)]
        )
    )
    if not identities_match:
        raise ValueError("raw episode identity/order/content changed; reviews cannot be reapplied")
    reviews = document.get("episodes")
    if not isinstance(reviews, dict):
        raise ValueError("invalid raw reviews")
    from ..modeling.dataset_trim import frame_bounds

    for index, review in reviews.items():
        if (not str(index).isdigit() or not 0 <= int(index) < len(stored_ids)
                or not isinstance(review, dict)):
            raise ValueError("invalid raw review episode")
        if review.get("status", "") not in {"", "keep", "review", "reject"}:
            raise ValueError("invalid raw review status")
        episode = episodes[int(index)]
        frame_bounds(review.get("trim"), int(episode.metadata["frames"]),
                     float(episode.session["record_fps"]))
    return reviews


def select_reviewed_episodes(episodes: list, reviews: dict) -> list[dict]:
    from ..modeling.dataset_trim import frame_bounds

    result = []
    for index, episode in enumerate(episodes):
        review = reviews.get(str(index), {})
        if review.get("status") in {"review", "reject"}:
            continue
        first, stop = frame_bounds(review.get("trim"), int(episode.metadata["frames"]),
                                   float(episode.session["record_fps"]))
        result.append({"episode": episode, "source_index": index,
                       "episode_id": episode_identity(episode),
                       "start_frame": first, "end_frame": stop,
                       "review": review})
    return result
