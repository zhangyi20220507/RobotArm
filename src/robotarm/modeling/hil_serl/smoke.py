"""Exercise learner, durable replay, inference bounds and resume without hardware."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import Config, atomic_json
from .learner import learn
from .replay import records_to_episode, write_episode
from .sac import load_actor


def smoke(args):
    run = args.output.resolve()
    if run.exists():
        raise FileExistsError("smoke output must be a new directory")
    if args.steps < 2:
        raise ValueError("smoke test needs at least two updates")
    run.mkdir(parents=True)
    cfg = Config("synthetic-no-hardware", "synthetic", 30, ["global", "wrist"],
                 hidden_dim=32, batch_size=16, replay_capacity=256, task="synthetic test")
    cfg.save(run / "config.json")
    rng = np.random.default_rng(1000)
    for episode in range(3):
        records = []
        for step in range(8):
            observation = rng.normal(0, 0.1, 30).astype(np.float32)
            observation[-7:] = 0
            records.append({"observation": observation, "action": np.zeros(7, np.float32),
                            "expert": episode == 0 or step == 4, "command_sent": True})
        outcome = "failure" if episode == 1 else "success"
        write_episode(run / "seed" / f"episode_{episode:06d}.npz",
                      records_to_episode(records, outcome, cfg), cfg,
                      metadata={"kind": "synthetic", "outcome": outcome})
    options = dict(run_dir=run, device="cpu", steps=args.steps, offline_steps=args.steps,
                   updates_per_transition=1, publish_every=2, save_every=5, cpu_threads=2, resume=False)
    learn(SimpleNamespace(**options))
    actor, version = load_actor(run / "policy.pt", cfg)
    state = torch.zeros(1, cfg.observation_dim)
    with torch.inference_mode():
        for _ in range(10):
            action, _ = actor(state)
            if not torch.isfinite(action).all() or (action.abs() > actor.limit + 1e-7).any():
                raise AssertionError("actor exceeded residual limits")
    options.update(resume=True, steps=args.steps + 1, offline_steps=args.steps + 1)
    learn(SimpleNamespace(**options))
    _, resumed_version = load_actor(run / "policy.pt", cfg)
    if resumed_version != version + 1:
        raise AssertionError("learner did not resume at the saved optimization step")
    result = {"passed": True, "updates": resumed_version, "hardware_connected": False,
              "checks": ["SAC updates", "balanced replay", "atomic policy sync", "residual bounds", "optimizer resume"]}
    atomic_json(run / "smoke_result.json", result)
    print(result)
    return 0
