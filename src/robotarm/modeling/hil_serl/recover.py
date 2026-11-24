"""Recover a completed RAW trial after an interrupted outcome/spool submission."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config, checkpoint_hash
from .prior import ACTPrior
from .replay import read_episode, records_to_episode, write_episode


def recover(args):
    run = args.run_dir.resolve()
    episode = args.raw_episode.resolve()
    root = episode.parent.parent
    cfg = Config.load(run / "config.json")
    session = json.loads((root / "session.json").read_text())
    metadata = json.loads((episode / "episode.json").read_text())
    if episode.parent.name != "episodes" or metadata.get("status") != "complete" or int(metadata.get("frames", 0)) < 1:
        raise ValueError("recovery requires a completed, non-empty RAW episode")
    if session.get("robot", {}).get("hil_config_signature") != cfg.signature:
        raise ValueError("RAW trial belongs to another HIL run / ACT prior")
    if metadata.get("task") != cfg.task:
        raise ValueError("RAW trial task differs from the HIL run")
    if args.outcome not in ("success", "failure"):
        raise ValueError("explicit success/failure outcome required")
    # A byte-preserving raw merge changes the directory and episode number,
    # but must not create a second replay submission for the original trial.
    origin = metadata.get("merge_source", {}).get("episode_path", str(episode))
    raw_identity = str(Path(origin).expanduser().resolve())
    identity = hashlib.sha256(raw_identity.encode()).hexdigest()
    destination = run / "online" / f"episode_{identity}.npz"
    if destination.exists():
        _, existing = read_episode(destination, cfg)
        if existing.get("raw_episode") != raw_identity or existing.get("outcome") != args.outcome:
            raise ValueError("existing replay submission conflicts with the requested recovery")
        print(f"Already submitted; no duplicate created: {destination}")
        return 0
    from ..raw_dataset_browser import RawDatasetStore
    store = RawDatasetStore(root)
    source_index = next((i for i, source in enumerate(store.sources) if source.path.resolve() == episode), None)
    if source_index is None:
        raise ValueError("completed RAW trial is not discoverable")
    review = store.reviews().get(str(source_index), {})
    has_review = review.get("review_origin") == "manual" and bool(review.get("status"))
    if has_review and (review.get("status") != "keep" or review.get("labels", {}).get("task_outcome") != args.outcome):
        raise ValueError("existing manual review conflicts with recovery; resolve it in the dataset browser")
    print(f"Recover {episode} as {args.outcome} -> {destination}")
    if not args.execute:
        print("PREVIEW: no annotations or replay written. Add --execute to recover; no hardware is used.")
        return 0
    if checkpoint_hash(cfg.act_checkpoint) != cfg.act_sha256:
        raise ValueError("ACT checkpoint changed; refusing incompatible replay reconstruction")
    import cv2
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    prior = ACTPrior.load(cfg.act_checkpoint, device=args.device, pool_size=cfg.pool_size,
                          gripper_max_m=cfg.gripper_max_m,
                          temporal_ensemble_coeff=cfg.temporal_ensemble_coeff)
    captures = {}
    try:
        if prior.roles != cfg.camera_roles:
            raise ValueError("ACT camera roles differ from the HIL run")
        captures = {role: cv2.VideoCapture(str(episode / f"{role}.mp4")) for role in cfg.camera_roles}
        if not all(cap.isOpened() for cap in captures.values()):
            raise ValueError("RAW camera video cannot be opened")
        records = []
        with np.load(episode / "frames.npz", allow_pickle=False) as data:
            n = int(metadata["frames"])
            if cfg.format_version.endswith("v2") and "hil_base_action" not in data:
                raise ValueError("v2 RAW recovery requires the recorded ACT reference for exact temporal/hold history")
            for key in ("observation_state", "action", "command_sent", "expert_mask"):
                if len(data[key]) != n:
                    raise ValueError("RAW frame count mismatch")
            for i in range(n):
                batch = {"observation.state": torch.from_numpy(data["observation_state"][i].astype(np.float32)).reshape(1, 7)}
                for role, cap in captures.items():
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"RAW {role} video ended before frame {i}")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    batch["observation.images." + role] = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
                reference = data["hil_base_action"][i] if cfg.format_version.endswith("v2") else None
                observation, _ = prior.encode(batch, reference_action=reference)
                records.append({"observation": observation, "action": data["action"][i].copy(),
                                "command_sent": bool(data["command_sent"][i]), "expert": bool(data["expert_mask"][i])})
        arrays = records_to_episode(records, args.outcome, cfg)
        if not has_review:
            from ..manual_hil_outcome import save_manual_outcome
            save_manual_outcome(root, episode, args.outcome)
        write_episode(destination, arrays, cfg, metadata={
            "kind": "online", "raw_episode": raw_identity, "outcome": args.outcome,
            "recovered_raw_path": str(episode),
            "policy_version": None, "act_sha256": cfg.act_sha256, "recovered_from_raw": True,
        })
        print(f"Recovered {len(arrays['state'])} executed transitions; learner will ingest automatically.")
        return 0
    finally:
        for cap in captures.values():
            cap.release()
        prior.close()
