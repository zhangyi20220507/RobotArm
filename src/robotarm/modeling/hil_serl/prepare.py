"""Cache frozen ACT features from a reviewed HIL replay dataset, without hardware."""
import json
from pathlib import Path
import tempfile

import numpy as np

from .config import Config, checkpoint_hash
from .prior import ACTPrior, dataset_batch
from .replay import records_to_episode, write_episode


def encoded_samples(dataset, prior, batch_size):
    """Encode temporal references sequentially; batch only legacy stateless references."""
    if batch_size < 1:
        raise ValueError("ACT preparation batch size must be positive")
    if isinstance(getattr(prior, "temporal_ensemble_coeff", None), (float, int)):
        previous_context = None
        for index in range(len(dataset)):
            sample = dataset[index]
            source = int(sample.get("complementary_info.control_source",
                                    2 if bool(sample.get("complementary_info.is_intervention", False)) else 0))
            context = (int(sample["episode_index"]), source != 0)
            if context != previous_context:
                prior.reset()
            previous_context = context
            yield sample, prior.encode(dataset_batch(sample, prior.roles))[0]
        return
    if batch_size == 1:
        for index in range(len(dataset)):
            sample = dataset[index]
            yield sample, prior.encode(dataset_batch(sample, prior.roles))[0]
        return
    import torch
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batches = [dataset_batch(sample, prior.roles) for sample in samples]
        inputs = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        observations, _ = prior.encode_batch(inputs)
        if len(observations) != len(samples):
            raise ValueError("ACT encoded batch size mismatch")
        yield from zip(samples, observations)


def inspect_dataset(root):
    root = Path(root).resolve()
    if (root / "CONVERSION_INCOMPLETE.txt").exists():
        raise ValueError("HIL replay conversion is incomplete")
    manifest = json.loads((root / "meta/hil_serl_conversion.json").read_text())
    if manifest["representation"] != {"space": "joint", "frame": "absolute"}:
        raise ValueError("Piper residual SAC requires joint / absolute HIL replay")
    if int(manifest.get("success_tail_frames", 1)) != 1:
        raise ValueError("export sparse RL rewards with --success-tail-frames 1 before preparing SAC")
    tasks = {episode["task"] for episode in manifest["episodes"]}
    if len(tasks) != 1 or not next(iter(tasks)):
        raise ValueError("one non-empty task is required per HIL run")
    info = json.loads((root / "meta/info.json").read_text())
    required = {"observation.state", "action", "next.reward", "next.done",
                "complementary_info.is_intervention", "complementary_info.command_sent"}
    if not required <= set(info["features"]):
        raise ValueError("dataset is missing HIL reward/done/control fields")
    return root, manifest, info, next(iter(tasks))


def prepare(args):
    root, manifest, info, task = inspect_dataset(args.dataset)
    checkpoint = args.act_checkpoint.resolve()
    run = args.run_dir.resolve()
    if run.exists():
        raise FileExistsError(f"use a new HIL run directory: {run}")
    print(f"ACT: {checkpoint}\nHIL replay: {root}\nRun: {run}")
    print(f"Episodes: {info['total_episodes']}; frames: {info['total_frames']}; task: {task}")
    if not args.execute:
        print("PREVIEW: no encoding or training. Add --execute to cache ACT features.")
        return 0
    import torch
    from lerobot.datasets import LeRobotDataset

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly for offline preparation")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    prior = ACTPrior.load(checkpoint, device=args.device,
                          gripper_max_m=args.gripper_max_mm / 1000,
                          temporal_ensemble_coeff=getattr(args, "temporal_ensemble_coeff", 0.01))
    print("HIL v2: sequential ACT select_action with temporal ensembling; new cache/run required.")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, download_videos=False, video_backend="pyav")
    run.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=run.name + ".prepare-", dir=run.parent) as temporary:
            stage = Path(temporary) / "run"
            stage.mkdir()
            cfg = None
            records = []
            current_episode = None
            expected_frame = 0
            episode_count = 0
            for sample, observation in encoded_samples(dataset, prior, getattr(args, "batch_size", 1)):
                episode = int(sample["episode_index"])
                frame = int(sample["frame_index"])
                if current_episode is not None and episode != current_episode:
                    raise ValueError("dataset crossed an episode boundary without terminal done")
                if frame != expected_frame:
                    raise ValueError("non-contiguous replay frame order")
                current_episode = episode
                expected_frame += 1
                if cfg is None:
                    cfg = Config(str(checkpoint), checkpoint_hash(checkpoint), len(observation), prior.roles,
                                 task=task, max_joint_delta_deg=args.max_joint_delta_deg,
                                 max_gripper_delta_mm=args.max_gripper_delta_mm,
                                 gripper_max_m=args.gripper_max_mm / 1000, seed=args.seed,
                                 rate_hz=float(info.get("fps", 20)),
                                 format_version="piper_hil_residual_sac_v2",
                                 temporal_ensemble_coeff=getattr(args, "temporal_ensemble_coeff", 0.01)).validate()
                    cameras = manifest.get("cameras", {})
                    if not cameras:
                        sources = manifest.get("source_sessions", [])
                        if sources and (Path(sources[0]) / "session.json").is_file():
                            cameras = json.loads((Path(sources[0]) / "session.json").read_text()).get("cameras", {})
                    cfg.camera_serials = {role: str(cameras[role]["serial"]) for role in cfg.camera_roles if role in cameras}
                    cfg.save(stage / "config.json")
                records.append({"observation": observation,
                                "action": sample["action"].detach().cpu().numpy().reshape(7),
                                "command_sent": bool(sample["complementary_info.command_sent"]),
                                "expert": bool(sample["complementary_info.is_intervention"])})
                reward = float(sample["next.reward"])
                done = bool(sample["next.done"])
                if reward not in (0, 1) or (reward and not done):
                    raise ValueError("only terminal sparse rewards are supported; export with --success-tail-frames 1")
                if done:
                    data = records_to_episode(records, "success" if reward else "failure", cfg)
                    write_episode(stage / "seed" / f"episode_{episode_count:06d}.npz", data, cfg,
                                  metadata={"source_dataset": str(root), "source_episode": episode,
                                            "outcome": "success" if reward else "failure", "kind": "seed"})
                    episode_count += 1
                    print(f"Cached episode {episode_count}/{info['total_episodes']} ({len(records)} frames)", flush=True)
                    records = []
                    current_episode = None
                    expected_frame = 0
            if records or cfg is None or episode_count != info["total_episodes"]:
                raise ValueError("replay dataset has incomplete or missing episodes")
            if cfg.act_sha256 != checkpoint_hash(checkpoint):
                raise RuntimeError("ACT checkpoint changed during feature preparation")
            (stage / "online").mkdir()
            stage.rename(run)
    finally:
        prior.close()
    print(f"Prepared {episode_count} episodes. Start hil-serl learner with --run-dir {run}")
    return 0
