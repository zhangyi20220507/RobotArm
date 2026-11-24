"""Local asynchronous learner with durable episode spool and atomic snapshots."""
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config, atomic_json
from .replay import Replay
from .sac import SAC


def learn(args):
    run = args.run_dir.resolve()
    cfg = Config.load(run / "config.json")
    if args.steps < 1 or args.publish_every < 1 or args.save_every < 1:
        raise ValueError("step and checkpoint intervals must be positive")
    if args.offline_steps < 1 or args.updates_per_transition < 1 or args.cpu_threads < 1:
        raise ValueError("offline steps, update ratio and CPU threads must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    torch.set_num_threads(min(torch.get_num_threads(), args.cpu_threads))
    lock = (run / "learner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another learner already owns this HIL run") from None
    try:
        torch.manual_seed(cfg.seed)
        algorithm = SAC(cfg, args.device)
        replay = Replay(cfg)
        replay.ingest(run / "seed")
        replay.ingest(run / "online")
        if not replay.count or not replay.human_count:
            raise ValueError("seed needs replay data and expert intervention / BC transitions")
        if args.resume:
            extra = algorithm.restore(run / "training.pt")
            if "replay_rng" in extra:
                replay.rng.bit_generator.state = extra["replay_rng"]
        elif (run / "training.pt").exists() or (run / "policy.pt").exists():
            raise FileExistsError("existing learner state; use --resume")

        def save():
            algorithm.save(run / "training.pt", training=True,
                           extra={"replay_rng": replay.rng.bit_generator.state})
            algorithm.save(run / "policy.pt")

        last_scan = 0.0
        waiting_budget = None
        print(f"Learner ready: replay={replay.count}, human={replay.human_count}, step={algorithm.step}", flush=True)
        try:
            with (run / "metrics.jsonl").open("a") as log:
                while algorithm.step < args.steps:
                    now = time.monotonic()
                    if now - last_scan >= 1:
                        added = replay.ingest(run / "online")
                        if added:
                            print(f"Ingested {added} online trials; replay={replay.count}, human={replay.human_count}", flush=True)
                        last_scan = now
                    budget = args.offline_steps + replay.online_count * args.updates_per_transition
                    if algorithm.step >= budget:
                        if waiting_budget != budget:
                            print(
                                f"[HIL] Update budget reached at step {algorithm.step}; "
                                "waiting for labelled online trials. Keep this learner running. "
                                "Start the HIL actor in another terminal; X/I/1/0 are actor keys, "
                                "not learner keys. Run directory: " + str(run),
                                flush=True,
                            )
                            waiting_budget = budget
                        time.sleep(0.1)
                        continue
                    metrics = algorithm.update(replay.sample())
                    if algorithm.step % args.publish_every == 0:
                        algorithm.save(run / "policy.pt")
                    if algorithm.step % args.save_every == 0:
                        save()
                    if algorithm.step == 1 or algorithm.step % 100 == 0:
                        metrics.update(time=time.time(), replay=replay.count, human=replay.human_count)
                        log.write(json.dumps(metrics) + "\n")
                        log.flush()
                        atomic_json(run / "learner_status.json", metrics)
                        print(json.dumps(metrics), flush=True)
                save()
        except KeyboardInterrupt:
            save()
            print("Learner interrupted; model, critics, optimizers and RNG saved.", flush=True)
        return 0
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
