"""Safely add Piper raw episodes to an existing LeRobot dataset.

The base dataset is copied to a staging directory and ``convert-piper-raw
--append`` runs only against that copy.  ``new`` publishes the result under a
new name.  ``append`` atomically replaces the base after success and retains a
complete sibling backup.  Dataset Browser reviews are remapped by keeping the
base indices and adding the new episodes after them.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..record.convert import (
    discover_episodes,
    read_json,
    split_expert_selections,
    uses_expert_mask,
    write_json,
)
from ..record.file_copy import clone_or_copy
from ..record.raw_reviews import load_raw_reviews, select_reviewed_episodes
from .dataset_trim import exported_reviews
from .quality_filter import default_reviews_path, validate_reviews


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _source_sessions(conversion: dict[str, Any]) -> list[str]:
    result: list[str] = []
    current: Any = conversion
    seen: set[int] = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        for value in current.get("source_sessions", []):
            resolved = str(Path(value).expanduser().resolve())
            if resolved not in result:
                result.append(resolved)
        current = current.get("source_conversion")
    return result


def _base_session(base: Path, conversion: dict[str, Any]) -> tuple[dict[str, Any], str]:
    direct = base / "meta" / "piper_raw_session.json"
    if direct.is_file():
        return read_json(direct), str(direct)

    trim_path = base / "meta" / "trim_provenance.json"
    if trim_path.is_file():
        trim = read_json(trim_path)
        source = Path(str(trim.get("source_dataset", ""))).expanduser()
        if not source.is_absolute():
            source = (base.parent / source).resolve()
        candidate = source / "meta" / "piper_raw_session.json"
        if candidate.is_file():
            return read_json(candidate), str(candidate)

    for session in _source_sessions(conversion):
        candidate = Path(session) / "session.json"
        if candidate.is_file():
            return read_json(candidate), str(candidate)
    raise RuntimeError(
        "cannot recover base piper_raw_session.json from the dataset, trim provenance, "
        "or conversion source sessions"
    )


def _load_reviews(base: Path, reviews_path: Path) -> dict[str, Any]:
    total = int(read_json(base / "meta" / "info.json")["total_episodes"])
    if reviews_path.is_file():
        document = read_json(reviews_path)
    else:
        document = exported_reviews(base) or {"dataset_root": str(base), "episodes": {}}
    recorded = document.get("dataset_root")
    if recorded and Path(recorded).expanduser().resolve() != base:
        raise RuntimeError(f"review file belongs to another dataset: {recorded}")
    validate_reviews(document, total)
    return document


def _extended_reviews(
    inherited: dict[str, Any], target: Path, first_new: int, total: int,
    *, status: str, label: str | None, note: str,
) -> dict[str, Any]:
    episodes = copy.deepcopy(inherited.get("episodes", {}))
    for index in range(first_new, total):
        review: dict[str, Any] = {"status": status, "note": note}
        if label is not None:
            review["labels"] = {"cup_orientation": label}
        episodes[str(index)] = review
    return {
        "dataset_root": str(target),
        "episodes": episodes,
        "derived_from": {
            "dataset_root": inherited.get("dataset_root"),
            "base_episode_count": first_new,
            "new_episode_range": [first_new, total],
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    write_json(temporary, value)
    os.replace(temporary, path)


def _validate_compatibility(
    base_session: dict[str, Any], conversion: dict[str, Any], raw_paths: list[Path]
) -> tuple[int, int, list[tuple[Path, dict[str, Any], int]]]:
    known = set(_source_sessions(conversion))
    duplicate = [str(path) for path in raw_paths if str(path) in known]
    if duplicate:
        raise RuntimeError("raw session already represented by the base dataset: " + ", ".join(duplicate))

    roles = list(conversion.get("camera_roles", []))
    if not roles:
        raise RuntimeError("base conversion metadata has no camera_roles")
    details = []
    episodes_total = frames_total = 0
    for path in raw_paths:
        session, episodes, skipped = discover_episodes(path, False)
        for key in ("record_fps", "state_names", "state_units", "action_semantics"):
            if session.get(key) != base_session.get(key):
                raise RuntimeError(f"{path}: incompatible {key}")
        if list(session.get("cameras", {})) != list(base_session.get("cameras", {})):
            raise RuntimeError(f"{path}: incompatible camera roles/order")
        recorded_roles = list(episodes[0].metadata.get("camera_roles", []))
        if not set(roles).issubset(recorded_roles):
            raise RuntimeError(f"{path}: missing converted camera role; base={roles}, raw={recorded_roles}")
        for role in roles:
            for key in ("serial", "stream"):
                if session["cameras"][role].get(key) != base_session["cameras"][role].get(key):
                    raise RuntimeError(f"{path}: incompatible camera {role} {key}")
        reviews = load_raw_reviews(path, episodes)
        selections = select_reviewed_episodes(episodes, reviews)
        if uses_expert_mask(session):
            selections = split_expert_selections(selections)
        if not selections:
            raise RuntimeError(f"{path}: no reviewed expert training ranges to append")
        frames = sum(
            int(item["end_frame"]) - int(item["start_frame"])
            for item in selections
        )
        episodes_total += len(selections)
        frames_total += frames
        details.append((path, session, len(selections)))
        for reason in skipped:
            print(f"Skipping {path.name}/{reason}")
    return episodes_total, frames_total, details


def _prepare_stage(
    stage: Path, base: Path, target: Path, base_session: dict[str, Any],
    conversion: dict[str, Any], session_source: str,
) -> dict[str, Any]:
    staged_conversion = copy.deepcopy(conversion)
    staged_conversion["repo_id"] = f"local/{target.name}"
    staged_conversion.setdefault("source_sessions", _source_sessions(conversion))
    staged_conversion["extension_base"] = str(base)
    write_json(stage / "meta" / "piper_conversion.json", staged_conversion)
    write_json(stage / "meta" / "piper_raw_session.json", base_session)

    trim_path = stage / "meta" / "trim_provenance.json"
    archived_trim = None
    if trim_path.is_file():
        archived = stage / "meta" / "base_trim_provenance.json"
        if archived.exists():
            raise RuntimeError(f"cannot archive trim provenance; file already exists: {archived}")
        trim_path.replace(archived)
        archived_trim = archived.name
    return {
        "base_dataset": str(base),
        "base_session_metadata_source": session_source,
        "base_conversion": conversion,
        "archived_trim_provenance": archived_trim,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="robotarm modeling extend-dataset --",
        description="Append Piper raw episodes transactionally or publish a new combined dataset.",
    )
    parser.add_argument("--mode", choices=("append", "new"), required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--new-raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="required for --mode new; forbidden for append")
    parser.add_argument("--base-reviews", type=Path, default=None)
    parser.add_argument("--output-reviews", type=Path, default=None)
    parser.add_argument("--new-status", choices=("review", "keep", "reject"), default="review")
    parser.add_argument(
        "--new-label", choices=("sideways", "upright", "uncertain", "none"), default="sideways"
    )
    parser.add_argument("--new-note", default="New appended episode; review before training")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="perform the copy/convert/publish operation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.base_dataset.expanduser().resolve()
    raw_paths = [path.expanduser().resolve() for path in args.new_raw]
    if args.mode == "new":
        if args.output is None:
            raise RuntimeError("--mode new requires --output")
        target = args.output.expanduser().resolve()
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"output already exists: {target}")
    else:
        if args.output is not None:
            raise RuntimeError("--mode append does not accept --output")
        target = base
    if not (base / "meta" / "info.json").is_file():
        raise RuntimeError(f"not a LeRobot dataset: {base}")
    if any((base / marker).exists() for marker in ("TRIM_INCOMPLETE.txt", "CONVERSION_INCOMPLETE.txt")):
        raise RuntimeError(f"base dataset is incomplete: {base}")
    if len(raw_paths) != len(set(raw_paths)):
        raise RuntimeError("duplicate --new-raw paths are not allowed")
    for raw in raw_paths:
        if not (raw / "session.json").is_file():
            raise RuntimeError(f"not a Piper raw session: {raw}")

    info = read_json(base / "meta" / "info.json")
    conversion = read_json(base / "meta" / "piper_conversion.json")
    base_session, session_source = _base_session(base, conversion)
    base_reviews_path = (args.base_reviews.expanduser().resolve() if args.base_reviews
                         else default_reviews_path(base).resolve())
    inherited = _load_reviews(base, base_reviews_path)
    new_episodes, new_frames, details = _validate_compatibility(base_session, conversion, raw_paths)
    first_new = int(info["total_episodes"])
    expected_total = first_new + new_episodes
    output_reviews = (args.output_reviews.expanduser().resolve() if args.output_reviews else
                      default_reviews_path(target).resolve())
    if args.mode == "new" and output_reviews.exists():
        raise RuntimeError(f"output review file already exists: {output_reviews}")

    print(f"Mode: {args.mode}")
    print(f"Base: {base} ({first_new} episodes / {info['total_frames']} frames)")
    for path, session, count in details:
        print(f"New raw: {path} ({count} episodes; task={session.get('task')!r})")
    print(f"Result: {target} ({expected_total} episodes; +{new_frames} frames before read-back verification)")
    print(f"Reviews: {base_reviews_path} -> {output_reviews}")
    print(f"New reviews: status={args.new_status}, cup_orientation={args.new_label}")
    print("Base review indices and labels will be preserved exactly.")
    if not args.execute:
        print("DRY-RUN: no files written. Add --execute to run.")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.extend-{_timestamp()}-{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise RuntimeError(f"staging path already exists: {stage}")
    required = _tree_bytes(base) + sum(_tree_bytes(path) for path in raw_paths) + 512 * 1024**2
    free = shutil.disk_usage(target.parent).free
    if free < required:
        raise RuntimeError(
            f"insufficient free space for transactional extension: need about {required / 1024**3:.2f} GiB, "
            f"have {free / 1024**3:.2f} GiB"
        )

    print(f"Copying base dataset to staging area: {stage}")
    shutil.copytree(base, stage, copy_function=clone_or_copy)
    extension = _prepare_stage(stage, base, target, base_session, conversion, session_source)
    write_json(stage / "meta" / "base_quality_reviews.json", inherited)
    extension["base_reviews_snapshot"] = "meta/base_quality_reviews.json"
    representation = conversion.get("representation", {})
    command = [
        sys.executable, "-m", "robotarm.record.convert", *(str(path) for path in raw_paths),
        "--output", str(stage), "--repo-id", f"local/{target.name}", "--append",
        "--space", str(representation.get("space", "joint")),
        "--frame", str(representation.get("frame", "absolute")),
        "--image-writer-threads", str(args.image_writer_threads),
        "--encoder-threads", str(args.encoder_threads),
    ]
    print("Appending and validating new episodes in staging area...")
    result = subprocess.run(command, check=False)
    if result.returncode:
        print(f"Extension failed; diagnostic staging directory retained: {stage}", file=sys.stderr)
        return int(result.returncode)

    final_info = read_json(stage / "meta" / "info.json")
    if int(final_info["total_episodes"]) != expected_total:
        raise RuntimeError(
            f"append read-back mismatch: expected {expected_total} episodes, got {final_info['total_episodes']}; "
            f"staging retained at {stage}"
        )
    final_conversion = read_json(stage / "meta" / "piper_conversion.json")
    extension.update({
        "format": "robotarm_dataset_extend_v1",
        "mode": args.mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "new_raw_sessions": [str(path) for path in raw_paths],
        "base_episodes": first_new,
        "new_episodes": new_episodes,
        "result_episodes": expected_total,
        "base_reviews": str(base_reviews_path),
    })
    final_conversion["extension"] = extension
    write_json(stage / "meta" / "piper_conversion.json", final_conversion)
    write_json(stage / "meta" / "dataset_extension.json", extension)
    reviews = _extended_reviews(
        inherited, target, first_new, expected_total,
        status=args.new_status,
        label=None if args.new_label == "none" else args.new_label,
        note=args.new_note,
    )

    backup = None
    if args.mode == "new":
        os.replace(stage, target)
    else:
        backup = base.parent / f"{base.name}_backup_before_append_{_timestamp()}"
        if backup.exists() or backup.is_symlink():
            raise RuntimeError(f"backup path already exists: {backup}")
        os.replace(base, backup)
        try:
            os.replace(stage, base)
        except BaseException:
            os.replace(backup, base)
            raise
        # Keep the external Dataset Browser decisions with the backup as well;
        # labels added after the original trim export may exist only there.
        write_json(backup / "meta" / "quality_reviews_at_backup.json", inherited)
    _atomic_json(output_reviews, reviews)
    print(f"DONE: {target}")
    print(f"Quality reviews: {output_reviews}")
    if backup is not None:
        print(f"Original dataset backup: {backup}")
    print(f"Review new episode range {first_new}..{expected_total - 1} before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
