from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from robotarm.modeling.dataset_relabel import (
    TASK_STAT_COLUMNS,
    clone_dataset,
    inspect_dataset,
    relabel_dataset,
    verify_dataset,
)


class DatasetRelabelTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        video = root / "videos/observation.images.global/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"immutable-video")
        (root / "meta/info.json").write_text(json.dumps({
            "total_episodes": 2, "total_frames": 3, "total_tasks": 2,
        }))
        (root / "meta/stats.json").write_text(json.dumps({
            "task_index": {"min": [0.0], "max": [1.0], "count": [3]},
        }))
        pq.write_table(pa.table({
            "task_index": pa.array([0, 1], type=pa.int64()),
            "task": pa.array(["old-a", "old-b"]),
        }), root / "meta/tasks.parquet")
        episode_columns = {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "tasks": pa.array([["old-a"], ["old-b"]], type=pa.list_(pa.string())),
        }
        for name in TASK_STAT_COLUMNS:
            episode_columns[name] = pa.array(
                [[0.0], [1.0]], type=pa.list_(pa.float64())
            )
        pq.write_table(
            pa.table(episode_columns),
            root / "meta/episodes/chunk-000/file-000.parquet",
        )
        pq.write_table(pa.table({
            "task_index": pa.array([0, 0, 1], type=pa.int64()),
            "value": pa.array([1.0, 2.0, 3.0]),
        }), root / "data/chunk-000/file-000.parquet")
        (root / "meta/piper_conversion.json").write_text(json.dumps({
            "repo_id": "local/source", "source_conversion": {"task": "old-a"},
        }))
        (root / "meta/trim_provenance.json").write_text(json.dumps({
            "repo_id": "local/source", "output_dataset": str(root),
        }))

    def test_publish_keeps_source_and_unifies_all_authoritative_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            self.make_dataset(source)
            before = inspect_dataset(source)
            clone_dataset(source, output)
            relabel_dataset(output, "new-task", source, before)
            after = verify_dataset(output, "new-task")

            self.assertEqual(after["tasks"], {"new-task": 2})
            self.assertEqual(inspect_dataset(source)["tasks"], {
                "old-a": 1, "old-b": 1,
            })
            source_video = source / "videos/observation.images.global/chunk-000/file-000.mp4"
            output_video = output / "videos/observation.images.global/chunk-000/file-000.mp4"
            self.assertEqual(source_video.stat().st_ino, output_video.stat().st_ino)
            manifest = json.loads(
                (output / "meta/task_relabel.json").read_text()
            )
            self.assertEqual(manifest["previous_episode_tasks"], {
                "old-a": 1, "old-b": 1,
            })
            tasks = pd.read_parquet(output / "meta/tasks.parquet")
            self.assertEqual(tasks.index.name, "task")
            self.assertEqual(tasks.index.tolist(), ["new-task"])
            self.assertEqual(tasks["task_index"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
