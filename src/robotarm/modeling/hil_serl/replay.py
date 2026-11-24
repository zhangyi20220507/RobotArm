"""Atomic episode transport and balanced human/online transition sampling."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

FIELDS = ("state", "next_state", "action", "reward", "done", "expert")


def validate_episode(data, cfg):
    n = len(data["state"])
    if n < 1:
        raise ValueError("empty HIL episode")
    shapes = {"state": (n, cfg.observation_dim), "next_state": (n, cfg.observation_dim),
              "action": (n, 7), "reward": (n, 1), "done": (n, 1), "expert": (n, 1)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid HIL transition array: {key}")
    for key in ("reward", "done", "expert"):
        if not np.isin(data[key], [0, 1]).all():
            raise ValueError(f"{key} must be binary")
    if not data["done"][-1, 0] or data["done"][:-1].any():
        raise ValueError("an episode must end exactly once; no cross-episode transitions")
    if data["reward"][:-1].any():
        raise ValueError("only a terminal success receives sparse reward")


def write_episode(path, arrays, cfg, *, metadata):
    validate_episode(arrays, cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        # An open file prevents numpy from appending a second suffix.
        with temp.open("wb") as out:
            np.savez_compressed(out, **{k: arrays[k] for k in FIELDS},
                                metadata=np.asarray(json.dumps({**metadata, "signature": cfg.signature})))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_episode(path, cfg):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"].item()))
        if meta["signature"] != cfg.signature:
            raise ValueError(f"{path}: replay belongs to another ACT prior / run")
        data = {key: archive[key].copy() for key in FIELDS}
    validate_episode(data, cfg)
    return data, meta


def records_to_episode(records, outcome, cfg):
    """Use only commanded actions; next observations never cross a trial boundary."""
    if outcome not in ("success", "failure"):
        raise ValueError("explicit success/failure outcome required")
    rows = []
    for index, record in enumerate(records):
        if not record["command_sent"]:
            continue
        following = records[min(index + 1, len(records) - 1)]
        rows.append((record, following))
    if not rows:
        raise ValueError("trial contains no executed actions")
    n = len(rows)
    data = {
        "state": np.asarray([r["observation"] for r, _ in rows], dtype=np.float32),
        "next_state": np.asarray([s["observation"] for _, s in rows], dtype=np.float32),
        "action": np.asarray([r["action"] for r, _ in rows], dtype=np.float32) / np.asarray(cfg.action_scale, dtype=np.float32),
        "expert": np.asarray([[r["expert"]] for r, _ in rows], dtype=np.float32),
        "reward": np.zeros((n, 1), dtype=np.float32), "done": np.zeros((n, 1), dtype=np.float32),
    }
    # Tail observations may be non-commanding frames. The final executed action
    # still receives the trial result and the actual final observation.
    data["next_state"][-1] = records[-1]["observation"]
    data["done"][-1] = 1
    data["reward"][-1] = float(outcome == "success")
    validate_episode(data, cfg)
    return data


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = deque()
        self.count = 0
        self.seen = {}
        self.online_count = 0
        self._all = []
        self._human = []
        self.rng = np.random.default_rng(cfg.seed)

    def ingest(self, directory):
        added = 0
        for file in sorted(Path(directory).glob("episode_*.npz"), key=lambda p: (p.stat().st_mtime_ns, p.name)):
            key = str(file.resolve())
            if key in self.seen:
                continue
            data, _ = read_episode(file, self.cfg)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in self.seen.values():
                raise ValueError(f"duplicate replay episode: {file}")
            if len(data["state"]) > self.cfg.replay_capacity:
                raise ValueError("one episode exceeds replay capacity")
            self.chunks.append(data)
            self._human.extend((data, i) for i in range(len(data["state"])) if data["expert"][i, 0])
            self._human = self._human[-self.cfg.replay_capacity:]
            self.count += len(data["state"])
            if Path(directory).name == "online":
                self.online_count += len(data["state"])
            self.seen[key] = digest
            added += 1
            while self.count > self.cfg.replay_capacity:
                self.count -= len(self.chunks.popleft()["state"])
        if added:
            self._all = [(data, i) for data in self.chunks for i in range(len(data["state"]))]
        return added

    @property
    def human_count(self):
        return len(self._human)

    def sample(self):
        if not self._all or not self._human:
            raise ValueError("both replay and human intervention/demo transitions are required")
        human = self.cfg.batch_size // 2
        rows = [self._human[i] for i in self.rng.integers(len(self._human), size=human)]
        rows += [self._all[i] for i in self.rng.integers(len(self._all), size=self.cfg.batch_size - human)]
        return {key: np.stack([data[key][i] for data, i in rows]) for key in FIELDS}
