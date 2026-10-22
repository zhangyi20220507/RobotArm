#!/usr/bin/env python3
"""Normalize the Piper pick-up dataset to one task and one setup id.

This script updates both the converted LeRobot dataset and the four raw source
sessions used to build it. It intentionally leaves older derived datasets
untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..paths import LEROBOT_DATA_ROOT, PIPER_RAW_ROOT


TASK = "Pick up the cup and place it in the target container"
SETUP_ID = "table_left_v1"
SOURCE_SESSIONS = (
    "20260818_102916_698587",
    "20260819_111307_534154",
    "20260819_130402_354756",
    "20260819_170411_263427",
)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, path.stat().st_mode)
    temporary.replace(path)


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="snappy")
    os.chmod(temporary, path.stat().st_mode)
    temporary.replace(path)


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"missing expected parquet column {name!r}")
    return table.set_column(index, name, values)


def copy_backup(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing file to back up: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_backup(dataset_root: Path, raw_root: Path, backup_root: Path) -> None:
    if backup_root.exists():
        raise RuntimeError(f"backup destination already exists: {backup_root}")

    dataset_files = (
        Path("meta/info.json"),
        Path("meta/stats.json"),
        Path("meta/tasks.parquet"),
        Path("meta/piper_conversion.json"),
        Path("meta/piper_raw_session.json"),
        Path("meta/episodes/chunk-000/file-000.parquet"),
        Path("data/chunk-000/file-000.parquet"),
    )
    for relative in dataset_files:
        copy_backup(dataset_root / relative, backup_root / "lerobot" / relative)

    for session_name in SOURCE_SESSIONS:
        session_root = raw_root / session_name
        copy_backup(
            session_root / "session.json",
            backup_root / "raw" / session_name / "session.json",
        )
        episode_files = sorted((session_root / "episodes").glob("episode_*/episode.json"))
        if not episode_files:
            raise RuntimeError(f"no episode metadata found in {session_root}")
        for episode_file in episode_files:
            relative = episode_file.relative_to(session_root)
            copy_backup(
                episode_file,
                backup_root / "raw" / session_name / relative,
            )


def update_dataset(dataset_root: Path) -> None:
    info_path = dataset_root / "meta/info.json"
    info = load_json(info_path)
    if info.get("total_episodes") != 64 or info.get("total_frames") != 39334:
        raise RuntimeError("dataset size differs from the expected 64 episodes / 39334 frames")
    info["total_tasks"] = 1
    atomic_write_json(info_path, info)

    stats_path = dataset_root / "meta/stats.json"
    stats = load_json(stats_path)
    task_stats = stats.get("task_index")
    if not task_stats or task_stats.get("count") != [39334]:
        raise RuntimeError("unexpected global task_index statistics")
    for statistic in task_stats:
        if statistic != "count":
            task_stats[statistic] = [0.0]
    atomic_write_json(stats_path, stats)

    conversion_path = dataset_root / "meta/piper_conversion.json"
    conversion = load_json(conversion_path)
    episodes = conversion.get("episodes", [])
    if len(episodes) != 64:
        raise RuntimeError("conversion provenance does not contain exactly 64 episodes")
    for episode in episodes:
        episode["task"] = TASK
    atomic_write_json(conversion_path, conversion)

    raw_session_path = dataset_root / "meta/piper_raw_session.json"
    raw_session = load_json(raw_session_path)
    raw_session["task"] = TASK
    raw_session["setup_id"] = SETUP_ID
    merged = raw_session.get("merged_source_sessions", [])
    if len(merged) != len(SOURCE_SESSIONS):
        raise RuntimeError("unexpected merged_source_sessions metadata")
    for session in merged:
        session["setup_id"] = SETUP_ID
    atomic_write_json(raw_session_path, raw_session)

    tasks_path = dataset_root / "meta/tasks.parquet"
    tasks = pq.read_table(tasks_path)
    if tasks.num_rows != 2 or set(tasks.column_names) != {"task_index", "task"}:
        raise RuntimeError("unexpected tasks.parquet structure")
    tasks = tasks.slice(0, 1)
    tasks = replace_column(tasks, "task_index", pa.array([0], type=pa.int64()))
    tasks = replace_column(tasks, "task", pa.array([TASK], type=pa.string()))
    atomic_write_parquet(tasks_path, tasks)

    episode_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_table = pq.read_table(episode_path)
    if episode_table.num_rows != 64:
        raise RuntimeError("unexpected episode metadata row count")
    task_lists = pa.array([[TASK] for _ in range(episode_table.num_rows)], type=pa.list_(pa.string()))
    episode_table = replace_column(episode_table, "tasks", task_lists)
    zeros = pa.array([[0.0] for _ in range(episode_table.num_rows)], type=pa.list_(pa.float64()))
    for column in TASK_STAT_COLUMNS:
        episode_table = replace_column(episode_table, column, zeros)
    atomic_write_parquet(episode_path, episode_table)

    data_path = dataset_root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(data_path)
    if data.num_rows != 39334:
        raise RuntimeError("unexpected frame data row count")
    frame_task_indices = pa.array([0] * data.num_rows, type=pa.int64())
    data = replace_column(data, "task_index", frame_task_indices)
    atomic_write_parquet(data_path, data)


def update_raw_sessions(raw_root: Path) -> None:
    total_episodes = 0
    for session_name in SOURCE_SESSIONS:
        session_root = raw_root / session_name
        session_path = session_root / "session.json"
        session = load_json(session_path)
        session["task"] = TASK
        session["setup_id"] = SETUP_ID
        atomic_write_json(session_path, session)

        episode_files = sorted((session_root / "episodes").glob("episode_*/episode.json"))
        total_episodes += len(episode_files)
        for episode_path in episode_files:
            episode = load_json(episode_path)
            episode["task"] = TASK
            atomic_write_json(episode_path, episode)
    if total_episodes != 64:
        raise RuntimeError(f"expected 64 raw episode metadata files, found {total_episodes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=LEROBOT_DATA_ROOT / "piper_pick_up",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PIPER_RAW_ROOT,
    )
    parser.add_argument("--backup-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    backup_root = args.backup_root.expanduser().resolve()
    build_backup(dataset_root, raw_root, backup_root)
    update_dataset(dataset_root)
    update_raw_sessions(raw_root)
    print(f"Unified task: {TASK}")
    print(f"Unified setup id: {SETUP_ID}")
    print(f"Backup: {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
