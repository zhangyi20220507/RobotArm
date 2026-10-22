"""Select training episodes from Dataset Browser quality reviews."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ..paths import ARTIFACTS_ROOT


INCLUDED_STATUSES = {"", "keep"}
KNOWN_STATUSES = INCLUDED_STATUSES | {"review", "reject"}


def default_reviews_path(dataset_root: Path) -> Path:
    return ARTIFACTS_ROOT / "reports" / "dataset_quality" / f"{dataset_root.name}.json"


def dataset_total_episodes(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"not a LeRobot dataset (missing {info_path})")
    return int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])


def load_reviews(reviews_path: Path, total: int) -> tuple[dict, dict[int, str]]:
    """Read a review file and validate its index/status map against ``total``."""
    if not reviews_path.is_file():
        raise ValueError(f"quality-review file not found: {reviews_path}")
    document = json.loads(reviews_path.read_text(encoding="utf-8"))
    reviews = document.get("episodes", {})
    if not isinstance(reviews, dict):
        raise ValueError("quality-review 'episodes' must be an object")
    statuses: dict[int, str] = {}
    for raw_index, review in reviews.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid episode index in quality reviews: {raw_index!r}") from exc
        if not 0 <= index < total:
            raise ValueError(f"quality-review episode {index} is outside dataset range 0..{total - 1}")
        if not isinstance(review, dict):
            raise ValueError(f"quality-review episode {index} must be an object")
        status = str(review.get("status", ""))
        if status not in KNOWN_STATUSES:
            raise ValueError(f"quality-review episode {index} has unknown status {status!r}")
        statuses[index] = status
    return document, statuses


def derive_reviews(
    target_root: Path, source_path: Path, destination: Path, overwrite: bool = False
) -> dict:
    """Copy another dataset's review decisions onto ``target_root``.

    Used when a dataset is rebuilt from the same raw sessions with a different
    conversion setting (for example a camera subset): the episode indices still
    line up, so the human review work carries over.  Both datasets must have the
    same episode count — otherwise the indices would silently refer to different
    episodes, which is exactly what the ``dataset_root`` guard in
    ``select_training_episodes`` exists to prevent.
    """
    target_root = target_root.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    total = dataset_total_episodes(target_root)
    try:
        document, _ = load_reviews(source_path, total)
    except ValueError as exc:
        raise ValueError(
            f"cannot derive reviews onto {target_root.name} "
            f"({total} episodes): {exc}"
        ) from exc
    if any(review.get("trim") is not None for review in document.get("episodes", {}).values()):
        raise ValueError(
            "cannot derive reviews containing trim ranges: "
            "请先导出裁剪数据集，不能将原帧范围直接套用到重建数据上"
        )
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination} already exists; pass --overwrite to replace it")
    derived = {
        "dataset_root": str(target_root),
        "episodes": document.get("episodes", {}),
        "derived_from": {
            "path": str(source_path),
            "dataset_root": document.get("dataset_root"),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same atomic swap the Dataset Browser uses, so an interrupted derive never
    # leaves a half-written review file behind.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return derived


def select_training_episodes(
    dataset_root: Path, reviews_path: Path, shuffle_seed: int | None = None
) -> tuple[list[int], Counter[str]]:
    """Episodes to train on, in the order LeRobot should receive them.

    Order matters: LeRobot splits train/eval by holding out the *last*
    ``ceil(n * eval_split)`` episodes of the list it is given, so an ascending
    list yields a sequential tail holdout (the episodes collected last).  Pass
    ``shuffle_seed`` to shuffle deterministically and get a random holdout
    instead; the same seed always produces the same split.
    """
    dataset_root = dataset_root.expanduser().resolve()
    reviews_path = reviews_path.expanduser().resolve()
    total = dataset_total_episodes(dataset_root)
    document, statuses = load_reviews(reviews_path, total)
    recorded_root = document.get("dataset_root")
    if recorded_root and Path(recorded_root).expanduser().resolve() != dataset_root:
        raise ValueError(
            f"quality-review file belongs to {Path(recorded_root).expanduser().resolve()}, not {dataset_root}"
        )

    counts: Counter[str] = Counter()
    selected: list[int] = []
    for index in range(total):
        status = statuses.get(index, "")
        counts[status or "unreviewed"] += 1
        if status in INCLUDED_STATUSES:
            if document.get("episodes", {}).get(str(index), {}).get("trim") is not None:
                raise ValueError(
                    f"episode {index} has an unapplied trim; "
                    "请在 Dataset Browser 导出裁剪后的新数据集，并使用新目录训练；"
                    "--quality-filter 不能只靠 episode 列表应用时间裁剪"
                )
            selected.append(index)
    if not selected:
        raise ValueError("quality filter selected no episodes")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(selected)
    return selected, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument(
        "--derive-from",
        type=Path,
        default=None,
        help=(
            "copy the review decisions of another dataset onto --dataset-root "
            "and exit, instead of printing the selected episodes. Both "
            "datasets must have the same episode count."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="with --derive-from, replace an existing review file",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=(
            "shuffle the selected episode order with this seed. LeRobot holds "
            "out the LAST ceil(n*eval_split) episodes of the list it receives, "
            "so shuffling turns the sequential tail holdout into a random one. "
            "Same seed -> same split. Omit for the historical ascending order."
        ),
    )
    args = parser.parse_args(argv)
    if args.derive_from is not None:
        destination = default_reviews_path(args.dataset_root)
        try:
            derived = derive_reviews(
                args.dataset_root, args.derive_from, destination, args.overwrite
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        # Count through the normal selection path so the summary reflects what
        # training will actually pick (absent episodes count as unreviewed) and
        # the derived file is proven to pass the dataset_root guard.
        try:
            selected, counts = select_training_episodes(args.dataset_root, destination)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(f"Derived reviews for {args.dataset_root.name}")
        print(f"  source  : {args.derive_from.expanduser().resolve()}")
        print(
            f"  keep={counts['keep']}, unreviewed={counts['unreviewed']}, "
            f"review={counts['review']}, reject={counts['reject']}; "
            f"selected={len(selected)}"
        )
        print(f"  written : {destination}")
        return 0
    reviews_path = args.reviews or default_reviews_path(args.dataset_root)
    try:
        selected, counts = select_training_episodes(
            args.dataset_root, reviews_path, args.shuffle_seed
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(selected, separators=(",", ":")))
    # stdout above must stay a single JSON line: train.bash captures it verbatim
    # into --dataset.episodes=.  Everything else goes to stderr.
    order = f"shuffled(seed={args.shuffle_seed})" if args.shuffle_seed is not None else "ascending"
    print(
        "Quality filter: "
        f"keep={counts['keep']}, unreviewed={counts['unreviewed']}, "
        f"review={counts['review']}, reject={counts['reject']}; "
        f"selected={len(selected)}; order={order}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
