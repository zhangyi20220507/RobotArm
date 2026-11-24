#!/usr/bin/env python3
"""Prepare full Piper trials as an intervention-aware HIL-SERL replay dataset.

The source remains immutable.  Episode outcomes are read from the raw review
sidecar (``labels.task_outcome``), while every model/armed/expert frame remains
in the exported trajectory.  This is deliberately different from the
HG-DAgger supervised export, which keeps only expert frames.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import LEROBOT_DATA_ROOT
from . import convert
from .raw_reviews import episode_identity, load_raw_reviews


FORMAT = "robotarm_hil_serl_replay_v1"
OUTCOMES = {"success", "failure"}


def sample_hil_trials(selections, *, hg_success_count, bc_success_count, seed):
    """Keep every reviewed HG failure and sample successes by source type."""
    if hg_success_count < 0 or bc_success_count < 0:
        raise ValueError("success counts must be non-negative")
    groups = {"hg_failure": [], "hg_success": [], "bc_success": []}
    seen = set()
    for item in selections:
        if item["episode_id"] in seen:
            raise ValueError("duplicate source trial; do not provide both merged and original sessions")
        seen.add(item["episode_id"])
        hg = convert.uses_expert_mask(item["episode"].session)
        group = ("hg_" + item["outcome"]) if hg else "bc_" + item["outcome"]
        if group in groups:
            groups[group].append(item)
    # Sort before sampling so input session order does not alter the sample.
    rng = random.Random(seed)
    chosen = list(groups["hg_failure"])
    for group, count in (("hg_success", hg_success_count), ("bc_success", bc_success_count)):
        pool = sorted(groups[group], key=lambda item: item["episode_id"])
        chosen.extend(rng.sample(pool, min(count, len(pool))))
    if not chosen:
        raise ValueError("no trials remain after HIL outcome sampling")
    return sorted(chosen, key=lambda item: item["episode_id"])


def default_output(session: Path) -> Path:
    return LEROBOT_DATA_ROOT / f"hil_serl_{session.name}"


def default_repo_id(session: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", session.name).strip("-.")
    return f"local/piper-hil-serl-{safe or 'dataset'}"


def build_features(
    session: dict[str, Any], camera_roles: list[str], space: str
) -> dict[str, dict[str, Any]]:
    features = convert.build_features(session, camera_roles, space)
    features.update(
        {
            "next.reward": {"dtype": "float32", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
            "complementary_info.is_intervention": {
                "dtype": "bool",
                "shape": (1,),
            },
            "complementary_info.control_source": {
                "dtype": "int64",
                "shape": (1,),
            },
            "complementary_info.intervention_id": {
                "dtype": "int64",
                "shape": (1,),
            },
            "complementary_info.command_sent": {
                "dtype": "bool",
                "shape": (1,),
            },
            "complementary_info.novice_action": {
                "dtype": "float32",
                "shape": (7,),
            },
            "complementary_info.novice_action_valid": {
                "dtype": "bool",
                "shape": (1,),
            },
        }
    )
    return features


def selected_trials(
    root: Path, episodes: list[convert.EpisodeSource]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select explicitly kept and outcome-labelled full trials."""
    reviews = load_raw_reviews(root, episodes)
    selected: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        review = reviews.get(str(index), {})
        if review.get("status") != "keep":
            continue
        if review.get("trim"):
            raise ValueError(
                f"{episode.path.name}: HIL-SERL requires the full trial; "
                "remove the review trim before preparing replay data"
            )
        outcome = review.get("labels", {}).get("task_outcome")
        if outcome not in OUTCOMES:
            raise ValueError(
                f"{episode.path.name}: kept HIL trial must be labelled "
                "task_outcome=success or failure in the raw review page"
            )
        selected.append(
            {
                "episode": episode,
                "source_index": index,
                "episode_id": episode_identity(episode),
                "outcome": outcome,
                "review": review,
            }
        )
    if not selected:
        raise ValueError(
            "no HIL trials selected: mark usable episodes Keep and set "
            "Task outcome (HIL) to Success or Failure in the raw review page"
        )
    return selected, reviews


def _control_arrays(
    data: Any, rows: int, *, full_hg_trial: bool, episode_index: int
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    if full_hg_trial:
        source = np.asarray(data["control_source"], dtype=np.int64)
        intervention = np.asarray(data["expert_mask"], dtype=bool)
        intervention_id = np.asarray(data["intervention_id"], dtype=np.int64)
        command_sent = np.asarray(data["command_sent"], dtype=bool)
    else:
        # An ordinary leader-follower episode is a full human demonstration.
        source = np.full(rows, 2, dtype=np.int64)
        intervention = np.ones(rows, dtype=bool)
        intervention_id = np.full(rows, episode_index, dtype=np.int64)
        command_sent = np.ones(rows, dtype=bool)
    for name, value in (
        ("control_source", source),
        ("is_intervention", intervention),
        ("intervention_id", intervention_id),
        ("command_sent", command_sent),
    ):
        if value.shape != (rows,):
            raise RuntimeError(f"{name} has shape {value.shape}, expected {(rows,)}")
    return source, intervention, intervention_id, command_sent


def convert_trial(
    dataset: Any,
    item: dict[str, Any],
    camera_roles: list[str],
    *,
    space: str,
    frame: str,
    fk: Any,
    full_hg_trial: bool,
    success_tail_frames: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    episode = item["episode"]
    rows = int(episode.metadata["frames"])
    captures = {
        role: cv2.VideoCapture(str(episode.path / f"{role}.mp4"))
        for role in camera_roles
    }
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise RuntimeError(f"failed to open a video in {episode.path}")

    reward_start = rows
    if item["outcome"] == "success":
        reward_start = max(0, rows - max(1, success_tail_frames))
    try:
        with np.load(episode.path / "frames.npz", allow_pickle=False) as data:
            actions = convert.compute_action_array(data, rows, space, frame, fk)
            if space == "joint":
                states = np.asarray(data["observation_state"], dtype=np.float32)
            else:
                states = np.concatenate(
                    [
                        np.asarray(data["observation_tcp_pose"], dtype=np.float32),
                        np.asarray(data["observation_state"][:, 6:7], dtype=np.float32),
                    ],
                    axis=1,
                )
            source, intervention, intervention_id, command_sent = _control_arrays(
                data,
                rows,
                full_hg_trial=full_hg_trial,
                episode_index=int(episode.metadata["episode_index"]),
            )
            if full_hg_trial:
                novice_actions = np.asarray(data["novice_action"], dtype=np.float32)
                if novice_actions.shape != (rows, 7):
                    raise RuntimeError(
                        f"novice_action has shape {novice_actions.shape}, expected {(rows, 7)}"
                    )
                novice_valid = (source == 0) & command_sent & np.isfinite(novice_actions).all(axis=1)
            else:
                novice_actions = np.zeros((rows, 7), dtype=np.float32)
                novice_valid = np.zeros(rows, dtype=bool)
            for index in range(rows):
                frame_dict: dict[str, Any] = {
                    "observation.state": states[index],
                    "action": actions[index],
                    "next.reward": np.asarray(
                        [1.0 if index >= reward_start else 0.0], dtype=np.float32
                    ),
                    "next.done": np.asarray([index == rows - 1], dtype=bool),
                    "complementary_info.is_intervention": np.asarray(
                        [intervention[index]], dtype=bool
                    ),
                    "complementary_info.control_source": np.asarray(
                        [source[index]], dtype=np.int64
                    ),
                    "complementary_info.intervention_id": np.asarray(
                        [intervention_id[index]], dtype=np.int64
                    ),
                    "complementary_info.command_sent": np.asarray(
                        [command_sent[index]], dtype=bool
                    ),
                    "complementary_info.novice_action": (
                        novice_actions[index] if novice_valid[index]
                        else np.zeros(7, dtype=np.float32)
                    ),
                    "complementary_info.novice_action_valid": np.asarray(
                        [novice_valid[index]], dtype=bool
                    ),
                    "task": str(episode.metadata["task"]),
                }
                for role, capture in captures.items():
                    ok, image_bgr = capture.read()
                    if not ok or image_bgr is None:
                        raise RuntimeError(
                            f"{episode.path.name}:{role}.mp4 ended at frame {index}"
                        )
                    frame_dict[f"observation.images.{role}"] = np.ascontiguousarray(
                        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    )
                dataset.add_frame(frame_dict)
        dataset.save_episode()
    finally:
        for capture in captures.values():
            capture.release()

    return {
        "source_session": episode.session_path.name,
        "source_episode": episode.path.name,
        "source_episode_index": episode.metadata.get("episode_index"),
        "raw_episode_id": item["episode_id"],
        "frames": rows,
        "outcome": item["outcome"],
        "reward_one_frames": max(0, rows - reward_start),
        "intervention_frames": int(intervention.sum()),
        "interventions": len(
            {int(value) for value in intervention_id[intervention] if int(value) >= 0}
        ),
        "task": episode.metadata.get("task"),
        "review": item["review"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate RAW reviews and report selected transitions without encoding")
    parser.add_argument("--selection", choices=("all", "hg-failures-plus-success"), default="all")
    parser.add_argument("--hg-success-count", type=int, default=0,
                        help="maximum HG successes sampled with hg-failures-plus-success")
    parser.add_argument("--bc-success-count", type=int, default=0,
                        help="maximum BC successes sampled with hg-failures-plus-success")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--buffer-role",
        choices=("replay", "offline-demo"),
        default="replay",
        help=(
            "replay keeps full autonomous/intervention trials; offline-demo is "
            "only valid for ordinary full-human demonstration sessions"
        ),
    )
    parser.add_argument("--camera-role", action="append", default=None)
    parser.add_argument("--space", choices=("joint", "tcp"), default="tcp")
    parser.add_argument("--frame", choices=("absolute", "relative"), default="relative")
    parser.add_argument(
        "--success-tail-frames",
        type=int,
        default=1,
        help=(
            "number of final frames labelled reward=1 for successful trials; "
            "keep 1 for sparse RL reward, increase only for a deliberately "
            "held success-state reward-classifier dataset"
        ),
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=2)
    return parser.parse_args(argv)


def _check_compatibility(
    primary: dict[str, Any], candidate: dict[str, Any], path: Path
) -> None:
    for key in ("record_fps", "state_names", "state_units", "action_semantics"):
        if candidate.get(key) != primary.get(key):
            raise RuntimeError(f"{path}: incompatible {key}")
    if list(candidate.get("cameras", {})) != list(primary.get("cameras", {})):
        raise RuntimeError(f"{path}: incompatible camera roles")
    for role in primary.get("cameras", {}):
        for key in ("serial", "stream"):
            if candidate["cameras"][role].get(key) != primary["cameras"][role].get(key):
                raise RuntimeError(f"{path}: incompatible camera {role} {key}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.success_tail_frames < 1:
        raise ValueError("--success-tail-frames must be at least 1")
    if args.image_writer_threads < 1 or args.encoder_threads < 1:
        raise ValueError("writer/encoder thread counts must be positive")
    if args.hg_success_count < 0 or args.bc_success_count < 0:
        raise ValueError("success counts must be non-negative")
    if args.selection == "all" and (args.hg_success_count or args.bc_success_count):
        raise ValueError("success counts require --selection hg-failures-plus-success")
    if args.selection != "all" and args.buffer_role != "replay":
        raise ValueError("mixed HG/BC selection requires --buffer-role replay")

    roots = [path.expanduser().resolve() for path in args.sessions]
    discovered = [convert.discover_episodes(root, False) for root in roots]
    session = discovered[0][0]
    for root, (candidate, _, _) in zip(roots[1:], discovered[1:]):
        _check_compatibility(session, candidate, root)

    hg_flags = [convert.uses_expert_mask(candidate) for candidate, _, _ in discovered]
    if args.buffer_role == "offline-demo" and any(hg_flags):
        raise ValueError(
            "--buffer-role offline-demo refuses full HG trials because they contain "
            "autonomous novice actions; use ordinary leader-follower demonstrations "
            "or export these trials as --buffer-role replay"
        )

    selections: list[dict[str, Any]] = []
    reviews_snapshot: dict[str, Any] = {}
    for root, (_, episodes, skipped) in zip(roots, discovered):
        for reason in skipped:
            print(f"Skipping {root.name}/{reason}")
        selected, reviews = selected_trials(root, episodes)
        selections.extend(selected)
        reviews_snapshot[str(root)] = reviews

    if args.selection == "hg-failures-plus-success":
        selections = sample_hil_trials(
            selections, hg_success_count=args.hg_success_count,
            bc_success_count=args.bc_success_count, seed=args.seed,
        )
    counts = {"hg_failure": 0, "hg_success": 0, "bc_failure": 0, "bc_success": 0}
    for item in selections:
        source = "hg" if convert.uses_expert_mask(item["episode"].session) else "bc"
        counts[source + "_" + item["outcome"]] += 1
    print("Selected trials: " + ", ".join(f"{key}={value}" for key, value in counts.items()))

    episodes = [item["episode"] for item in selections]
    recorded_roles = list(episodes[0].metadata.get("camera_roles", []))
    camera_roles = convert.select_camera_roles(recorded_roles, args.camera_role)
    for episode in episodes:
        convert.validate_episode(episode, episode.session, recorded_roles)

    fk = convert.load_tcp_fk() if args.space == "tcp" else None
    if fk is not None:
        for episode in episodes:
            convert.validate_fk_consistency(
                episode, fk, int(episode.metadata["frames"])
            )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output(roots[0]).resolve()
    )
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("output must be separate from raw source directories")
    if args.dry_run:
        print(f"Output: {output}")
        print(f"Buffer: {args.buffer_role}; selected trials: {len(selections)}")
        print(f"Transitions: {sum(int(item['episode'].metadata['frames']) for item in selections)}")
        print(f"Outcomes: {sum(item['outcome'] == 'success' for item in selections)} success, "
              f"{sum(item['outcome'] == 'failure' for item in selections)} failure")
        print("DRY-RUN: no LeRobot dataset was written")
        return 0
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot 0.6.1 is required in the robotarm environment") from exc

    repo_id = args.repo_id or default_repo_id(roots[0])
    encoder = RGBEncoderConfig(vcodec="h264", crf=23, preset="fast")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=int(session["record_fps"]),
        robot_type="piper_hil_serl",
        features=build_features(session, camera_roles, args.space),
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        rgb_encoder=encoder,
        encoder_threads=args.encoder_threads,
    )
    marker = output / "CONVERSION_INCOMPLETE.txt"
    marker.write_text("HIL-SERL conversion in progress; do not train.\n", encoding="utf-8")
    converted: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(selections):
            episode = item["episode"]
            if episode_identity(episode) != item["episode_id"]:
                raise RuntimeError(f"raw source changed before conversion: {episode.path}")
            print(
                f"Converting {episode.session_path.name}/{episode.path.name} "
                f"({item['outcome']}) -> episode {index}...",
                flush=True,
            )
            converted.append(
                convert_trial(
                    dataset,
                    item,
                    camera_roles,
                    space=args.space,
                    frame=args.frame,
                    fk=fk,
                    full_hg_trial=convert.uses_expert_mask(episode.session),
                    success_tail_frames=args.success_tail_frames,
                )
            )
            if episode_identity(episode) != item["episode_id"]:
                raise RuntimeError(f"raw source changed during conversion: {episode.path}")
        dataset.finalize()
    except Exception:
        marker.write_text(
            "HIL-SERL conversion failed. Remove this output or rerun with --overwrite.\n",
            encoding="utf-8",
        )
        raise

    provenance = {
        "format_version": FORMAT,
        "prepared_at": datetime.now().astimezone().isoformat(),
        "source_format": convert.RAW_FORMAT,
        "source_sessions": [str(root) for root in roots],
        "cameras": session["cameras"],
        "repo_id": repo_id,
        "buffer_role": args.buffer_role,
        "selection": {"mode": args.selection, "hg_success_count": args.hg_success_count,
                      "bc_success_count": args.bc_success_count, "seed": args.seed,
                      "actual_counts": counts},
        "representation": {"space": args.space, "frame": args.frame},
        "reward_semantics": (
            "binary sparse reward; successful final N frames are 1, all other "
            "frames are 0"
        ),
        "success_tail_frames": args.success_tail_frames,
        "done_semantics": "true on the final frame of every trial",
        "intervention_semantics": (
            "HG: expert_mask; ordinary leader-follower demo: true on every frame"
        ),
        "episodes": converted,
        "raw_reviews_snapshot": reviews_snapshot,
    }
    convert.write_json(output / "meta" / "hil_serl_conversion.json", provenance)

    info = convert.read_json(output / "meta" / "info.json")
    expected_frames = sum(item["frames"] for item in converted)
    if int(info["total_frames"]) != expected_frames:
        raise RuntimeError(
            f"read-back frame count {info['total_frames']} != {expected_frames}"
        )
    if int(info["total_episodes"]) != len(converted):
        raise RuntimeError("read-back episode count mismatch")
    marker.unlink()
    print(
        f"Done: {len(converted)} full trials / {expected_frames} transitions -> {output}"
    )
    print(
        f"Outcomes: {sum(x['outcome'] == 'success' for x in converted)} success, "
        f"{sum(x['outcome'] == 'failure' for x in converted)} failure; "
        f"intervention frames: {sum(x['intervention_frames'] for x in converted)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"HIL-SERL preparation failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
