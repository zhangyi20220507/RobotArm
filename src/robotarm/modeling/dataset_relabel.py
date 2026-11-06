"""Publish a LeRobot dataset with every episode assigned to one task.

Video payloads are hard-linked into the new dataset and are never re-encoded.
All mutable metadata and frame parquet files are copied before they are edited,
so the source dataset remains unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TASK_STAT_COLUMNS = (
    "stats/task_index/min",
    "stats/task_index/max",
    "stats/task_index/mean",
    "stats/task_index/std",
    "stats/task_index/q01",
    "stats/task_index/q10",
    "stats/task_index/q50",
    "stats/task_index/q90",
    "stats/task_index/q99",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.relabel.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.relabel.tmp")
    pq.write_table(table, temporary, compression="snappy")
    temporary.replace(path)


def _replace(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"missing expected parquet column {name!r}")
    return table.set_column(index, name, values)


def inspect_dataset(root: Path) -> dict:
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.parquet"
    episode_files = sorted((root / "meta/episodes").glob("**/*.parquet"))
    data_files = sorted((root / "data").glob("**/*.parquet"))
    stats_path = root / "meta/stats.json"
    if not all(path.is_file() for path in (info_path, tasks_path, stats_path)):
        raise RuntimeError(f"not a complete LeRobot dataset: {root}")
    if not episode_files or not data_files:
        raise RuntimeError("dataset has no episode or frame parquet files")

    info = _json(info_path)
    counts: Counter[str] = Counter()
    episode_rows = 0
    for path in episode_files:
        table = pq.read_table(path, columns=["tasks"])
        episode_rows += table.num_rows
        for tasks in table.column("tasks").to_pylist():
            if not isinstance(tasks, list) or len(tasks) != 1 or not tasks[0]:
                raise RuntimeError(
                    f"episode must have exactly one non-empty task: {path}"
                )
            counts[str(tasks[0])] += 1
    frame_rows = sum(pq.read_metadata(path).num_rows for path in data_files)
    if episode_rows != info.get("total_episodes"):
        raise RuntimeError("episode parquet count disagrees with info.json")
    if frame_rows != info.get("total_frames"):
        raise RuntimeError("frame parquet count disagrees with info.json")
    return {
        "episodes": episode_rows,
        "frames": frame_rows,
        "tasks": dict(sorted(counts.items())),
        "episode_files": episode_files,
        "data_files": data_files,
    }


def clone_dataset(source: Path, output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise RuntimeError("output dataset cannot be inside the source dataset")

    def copy_file(src: str, dst: str) -> str:
        relative = Path(src).relative_to(source)
        if relative.parts and relative.parts[0] == "videos":
            try:
                os.link(src, dst)
            except OSError as error:
                raise RuntimeError(
                    "video hard-link failed; source and output must be on the "
                    "same filesystem"
                ) from error
            return dst
        return shutil.copy2(src, dst)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output, copy_function=copy_file)


def relabel_dataset(root: Path, task: str, source: Path, before: dict) -> None:
    info_path = root / "meta/info.json"
    info = _json(info_path)
    info["total_tasks"] = 1
    _atomic_json(info_path, info)

    stats_path = root / "meta/stats.json"
    stats = _json(stats_path)
    task_stats = stats.get("task_index")
    if not isinstance(task_stats, dict):
        raise RuntimeError("stats.json has no task_index statistics")
    for name in task_stats:
        if name != "count":
            task_stats[name] = [0.0]
    _atomic_json(stats_path, stats)

    # LeRobot 0.6.x resolves a frame task with
    # ``meta.tasks.iloc[task_index].name``.  The language string therefore has
    # to be the persisted pandas index, not an ordinary dataframe column.
    # Arrow still exposes the physical index field as ``task`` in parquet.
    task_frame = pd.DataFrame(
        {"task_index": pd.array([0], dtype="int64")},
        index=pd.Index([task], name="task"),
    )
    tasks = pa.Table.from_pandas(task_frame, preserve_index=True)
    _atomic_parquet(root / "meta/tasks.parquet", tasks)

    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        task_type = table.schema.field("tasks").type
        table = _replace(
            table,
            "tasks",
            pa.array([[task]] * table.num_rows, type=task_type),
        )
        for name in TASK_STAT_COLUMNS:
            if name in table.column_names:
                value_type = table.schema.field(name).type
                table = _replace(
                    table,
                    name,
                    pa.array([[0.0]] * table.num_rows, type=value_type),
                )
        _atomic_parquet(path, table)

    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path)
        task_type = table.schema.field("task_index").type
        table = _replace(
            table,
            "task_index",
            pa.array([0] * table.num_rows, type=task_type),
        )
        _atomic_parquet(path, table)

    repo_id = f"local/{root.name}"
    for name in ("piper_conversion.json", "trim_provenance.json"):
        path = root / "meta" / name
        if not path.is_file():
            continue
        value = _json(path)
        if "repo_id" in value:
            value["repo_id"] = repo_id
        if name == "trim_provenance.json" and "output_dataset" in value:
            value["output_dataset"] = str(root)
        _atomic_json(path, value)

    manifest = {
        "format": "robotarm_task_relabel_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "output_dataset": str(root),
        "task": task,
        "previous_episode_tasks": before["tasks"],
        "total_episodes": before["episodes"],
        "total_frames": before["frames"],
        "videos_reencoded": False,
        "videos_hardlinked": True,
    }
    _atomic_json(root / "meta/task_relabel.json", manifest)


def verify_dataset(root: Path, task: str) -> dict:
    after = inspect_dataset(root)
    if after["tasks"] != {task: after["episodes"]}:
        raise RuntimeError("episode task relabel verification failed")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if (tasks.index.name != "task" or tasks.index.tolist() != [task]
            or tasks.columns.tolist() != ["task_index"]
            or tasks["task_index"].tolist() != [0]):
        raise RuntimeError("tasks.parquet relabel verification failed")
    for path in after["data_files"]:
        values = pq.read_table(path, columns=["task_index"])["task_index"]
        if values.null_count or set(values.to_pylist()) != {0}:
            raise RuntimeError(f"frame task_index verification failed: {path}")
    return after


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="publish the relabeled dataset; otherwise only print the plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    task = args.task.strip()
    if not task:
        raise RuntimeError("task must be non-empty")
    before = inspect_dataset(source)
    plan = {
        "mode": "execute" if args.execute else "preview",
        "source": str(source),
        "output": str(output),
        "task": task,
        "episodes": before["episodes"],
        "frames": before["frames"],
        "previous_tasks": before["tasks"],
        "videos_reencoded": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Preview only; add --execute to publish the new dataset.")
        return 0

    clone_dataset(source, output)
    marker = output / "RELABEL_INCOMPLETE.txt"
    marker.write_text("Task relabel is incomplete; do not train from this dataset.\n")
    relabel_dataset(output, task, source, before)
    after = verify_dataset(output, task)
    marker.unlink()
    print(
        f"Relabeled dataset ready: {output} "
        f"({after['episodes']} episodes, {after['frames']} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
