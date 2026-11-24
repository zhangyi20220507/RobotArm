"""Browse immutable raw MP4/NPZ, curate once, convert selected frames once."""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import threading
from bisect import bisect_right

from ..paths import LEROBOT_DATA_ROOT
from ..record import convert
from ..record.raw_reviews import (
    default_raw_reviews_path, load_raw_reviews, review_document,
)
from .dataset_browser import DatasetStore, _UNSET, _episode_metrics


class RawDatasetStore(DatasetStore):
    def __init__(self, root: Path, reviews_path: Path | None = None,
                 export_root: Path | None = None) -> None:
        self.root = root.expanduser().resolve()
        if (self.root / "MERGE_INCOMPLETE.txt").exists():
            raise ValueError("raw merge is incomplete")
        self.session, self.sources, self.skipped = convert.discover_episodes(self.root, False)
        self.reviews_path = (reviews_path or default_raw_reviews_path(self.root)).expanduser().resolve()
        self._output_root = (export_root or LEROBOT_DATA_ROOT).expanduser().resolve()
        if self._output_root == self.root or self.root in self._output_root.parents:
            raise ValueError("LeRobot export root must be outside the raw dataset")
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._export = {"status": "idle"}
        self.episodes = [{"episode_index": index, "length": int(item.metadata["frames"]),
                          "tasks": [item.metadata["task"]]}
                         for index, item in enumerate(self.sources)]
        self.info = {"fps": self.session["record_fps"], "total_episodes": len(self.sources)}
        self._binding = review_document(self.root, self.sources, {})
        self.reviews()

    def validate_review_document(self, document: dict) -> None:
        if (self.root / "APPEND_INCOMPLETE.json").exists():
            raise ValueError("raw transactional append is incomplete; recover it before saving reviews")
        load_raw_reviews(self.root, self.sources, document=document or self._binding,
                         allow_append=True)

    def review_metadata(self) -> dict:
        return {key: value for key, value in self._binding.items() if key != "episodes"}

    @property
    def export_parent(self) -> Path:
        return self._output_root

    def meta_payload(self) -> dict:
        result = super().meta_payload()
        result.update({"source_format": "piper_raw_v1", "raw_reviews": True,
                       "skipped": self.skipped,
                       "suggested_export_name": self.root.name[:80] + "_curated_"
                       + datetime.now().strftime("%Y%m%d_%H%M%S")})
        return result

    def episode_payload(self, index: int) -> dict:
        import numpy as np

        if not 0 <= index < len(self.sources):
            raise ValueError("invalid episode index")
        episode = self.sources[index]
        path = (episode.path / "frames.npz").resolve()
        if self.root not in path.parents:
            raise ValueError("raw payload must stay within its dataset")
        with np.load(path, allow_pickle=False) as data:
            state = data["observation_state"].astype(float)
            action = data["action"].astype(float)
            length = int(episode.metadata["frames"])
            if state.shape != (length, 7) or action.shape != (length, 7):
                raise ValueError("raw frame count/state/action mismatch")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("raw state/action contains nonfinite values")
            control_source = (
                np.asarray(data["control_source"], dtype=np.int8)
                if "control_source" in data
                else np.zeros(length, dtype=np.int8)
            )
            expert_mask = (
                np.asarray(data["expert_mask"], dtype=np.int8)
                if "expert_mask" in data
                else np.zeros(length, dtype=np.int8)
            )
            intervention_id = (
                np.asarray(data["intervention_id"], dtype=np.int64)
                if "intervention_id" in data
                else np.full(length, -1, dtype=np.int64)
            )
        fps = float(self.info["fps"])
        roles = list(episode.metadata["camera_roles"])
        return {"index": index, "length": length, "duration": length / fps,
                "tasks": [episode.metadata["task"]], "names": self.session["state_names"],
                "timestamps": (np.arange(length) / fps).tolist(),
                "state": state.tolist(), "action": action.tolist(),
                "control_source": control_source.tolist(),
                "expert_mask": expert_mask.tolist(),
                "intervention_id": intervention_id.tolist(),
                "expert_frames": int(np.count_nonzero(expert_mask)),
                "interventions": len(
                    {int(value) for value in intervention_id if int(value) >= 0}
                ),
                "metrics": _episode_metrics(state, action),
                "source": episode.metadata.get("merge_source", {"episode_path": str(episode.path)}),
                "videos": [{"key": f"observation.images.{role}", "url": f"/media/{index}/{ordinal}",
                            "from": 0} for ordinal, role in enumerate(roles)]}

    def video_path(self, episode: int, ordinal: int) -> Path:
        source = self.sources[episode]
        role = source.metadata["camera_roles"][ordinal]
        path = (source.path / f"{role}.mp4").resolve()
        if not path.is_file() or self.root not in path.parents:
            raise FileNotFoundError(path)
        return path

    def export_selected(self, output, plan, reviews, progress):
        # Bind to the exact files reviewed and snapshot selection before work starts.
        document = copy.deepcopy(self._binding)
        selected = {item["source_episode"] for item in plan}
        document["episodes"] = copy.deepcopy(reviews)
        for index in range(len(self.sources)):
            if index not in selected:
                document["episodes"][str(index)] = {"status": "reject"}
        convert.main([str(self.root), "--output", str(output), "--repo-id", f"local/{output.name}"],
                     progress=progress, review_documents={self.root: document})
        info_path = output / "meta" / "info.json"
        info = (
            convert.read_json(info_path)
            if info_path.is_file()
            else {
                "total_episodes": len(plan),
                "total_frames": sum(item["frames"] for item in plan),
            }
        )
        return {
            "output": str(output),
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
        }


class RawCollectionStore(DatasetStore):
    """One review page over several immutable RAW sessions and their own sidecars."""

    def __init__(self, roots: list[Path], export_root: Path | None = None) -> None:
        if len(roots) != len(set(roots)):
            raise ValueError("duplicate RAW session")
        self.stores = [RawDatasetStore(root, export_root=export_root) for root in roots]
        if len({float(store.info["fps"]) for store in self.stores}) != 1:
            raise ValueError("RAW sessions must use the same FPS")
        self.offsets = []
        self.episodes = []
        for store in self.stores:
            self.offsets.append(len(self.episodes))
            for row in store.episodes:
                self.episodes.append({**row, "episode_index": len(self.episodes),
                                      "source_session": store.root.name})
        self.root = roots[0]
        self.reviews_path = roots[0] / "annotations/reviews.json"
        self._output_root = self.stores[0].export_parent
        self.info = {"fps": self.stores[0].info["fps"],
                     "total_episodes": len(self.episodes)}
        self._export_lock = threading.Lock()
        self._export = {"status": "idle"}

    def _source(self, index: int) -> tuple[RawDatasetStore, int]:
        if not 0 <= index < len(self.episodes):
            raise ValueError("invalid episode index")
        slot = bisect_right(self.offsets, index) - 1
        return self.stores[slot], index - self.offsets[slot]

    def reviews(self) -> dict:
        return {str(offset + int(index)): review
                for offset, store in zip(self.offsets, self.stores)
                for index, review in store.reviews().items()}

    def save_review(self, episode, status, note, trim=_UNSET, labels=_UNSET):
        store, local = self._source(episode)
        kwargs = {}
        if trim is not _UNSET:
            kwargs["trim"] = trim
        if labels is not _UNSET:
            kwargs["labels"] = labels
        store.save_review(local, status, note, **kwargs)
        return self.effective_reviews()

    def episode_payload(self, index: int) -> dict:
        store, local = self._source(index)
        payload = store.episode_payload(local)
        payload["index"] = index
        payload["source_session"] = str(store.root)
        for video in payload["videos"]:
            video["url"] = video["url"].replace(f"/media/{local}/", f"/media/{index}/")
        return payload

    def video_path(self, episode: int, ordinal: int) -> Path:
        store, local = self._source(episode)
        return store.video_path(local, ordinal)

    @property
    def export_parent(self) -> Path:
        return self._output_root

    def meta_payload(self) -> dict:
        payload = super().meta_payload()
        for entry, row in zip(payload["episodes"], self.episodes):
            entry["source_session"] = row["source_session"]
        payload.update({"source_format": "piper_raw_v1", "raw_reviews": True,
                        "source_sessions": [str(store.root) for store in self.stores],
                        "reviews_paths": [str(store.reviews_path) for store in self.stores]})
        return payload

    def export_selected(self, output, plan, reviews, progress):
        """Export each RAW source separately, preserving BC/HG label semantics."""
        results = []
        destinations = [output.parent / f"{output.name}_{store.root.name}" for store in self.stores]
        if len(destinations) != len(set(destinations)) or any(path.exists() for path in destinations):
            raise FileExistsError("export destination exists or RAW session names repeat")
        for offset, store in zip(self.offsets, self.stores):
            local_plan = [{**item, "source_episode": item["source_episode"] - offset}
                          for item in plan if offset <= item["source_episode"] < offset + len(store.episodes)]
            if not local_plan:
                continue
            local_reviews = {str(int(index) - offset): review for index, review in reviews.items()
                             if offset <= int(index) < offset + len(store.episodes)}
            destination = output.parent / f"{output.name}_{store.root.name}"
            if destination.exists():
                raise FileExistsError(destination)
            results.append(store.export_selected(destination, local_plan, local_reviews, progress))
        return {"output": str(output.parent), "datasets": results,
                "episodes": sum(item["episodes"] for item in results),
                "frames": sum(item["frames"] for item in results)}
