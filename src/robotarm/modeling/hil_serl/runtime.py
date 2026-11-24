"""Actor adapter invoked inside the existing guarded Piper online control loop."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import Config, checkpoint_hash
from .prior import ACTPrior
from .replay import records_to_episode, write_episode
from .sac import load_actor


class HILRuntime:
    def __init__(self, run, checkpoint, policy, preprocessor, postprocessor, *,
                 device="cpu", deterministic=True, minimum_updates=100):
        self.run = Path(run).resolve()
        self.cfg = Config.load(self.run / "config.json")
        if minimum_updates < 1:
            raise ValueError("at least one completed learner update is required")
        if checkpoint_hash(checkpoint) != self.cfg.act_sha256:
            raise ValueError("HIL run is bound to a different ACT checkpoint / processors")
        self.prior = ACTPrior(policy, preprocessor, postprocessor,
                              pool_size=self.cfg.pool_size, gripper_max_m=self.cfg.gripper_max_m,
                              temporal_ensemble_coeff=self.cfg.temporal_ensemble_coeff)
        if self.prior.roles != self.cfg.camera_roles:
            self.prior.close()
            raise ValueError("HIL camera roles/order differ from the prepared ACT features")
        self.device = device
        self.deterministic = deterministic
        self.minimum_updates = minimum_updates
        self.version = -1
        self.records = None
        self.pending = {}
        self.current_observation = None
        self.base_action = None
        self.actor = None
        self._snapshot_token = None
        try:
            self.reload()
        except Exception:
            self.prior.close()
            raise

    def reload(self):
        if self.records is not None:
            raise RuntimeError("policy synchronization is only allowed between trials")
        path = self.run / "policy.pt"
        stat = path.stat()
        token = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self.actor is not None and token == self._snapshot_token:
            print(f"[HIL] using learner step {self.version} (weights unchanged)", flush=True)
            return
        candidate, version = load_actor(path, self.cfg, self.device, self.minimum_updates)
        if version < self.version:
            raise ValueError("refusing to roll back the actor to an older learner version")
        # Pay CUDA/head first-call cost while paused. The synthetic observation
        # is never executed or recorded; warm the deterministic path to avoid
        # consuming exploration randomness.
        with torch.inference_mode():
            candidate(torch.zeros(1, self.cfg.observation_dim, device=self.device), deterministic=True)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
        self.actor, self.version = candidate, version
        after = path.stat()
        after_token = (after.st_ino, after.st_size, after.st_mtime_ns)
        self._snapshot_token = token if token == after_token else None
        print(f"[HIL] synchronized learner step {version}", flush=True)

    def start_trial(self):
        if self.pending:
            raise RuntimeError("label the previous HIL trial before starting another")
        self.reload()
        self.prior.reset()
        self.records = []
        self.current_observation = None

    @torch.inference_mode()
    def predict(self, batch, *, autonomous=True, held_action=None):
        if held_action is None:
            observation, base = self.prior.encode(batch)
        else:
            observation, base = self.prior.encode(batch, reference_action=held_action)
        if observation.shape != (self.cfg.observation_dim,):
            raise ValueError("HIL observation shape changed")
        self.current_observation, self.base_action = observation, base
        if not autonomous or held_action is not None:
            return base.copy()
        tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        normalized, _ = self.actor(tensor, deterministic=self.deterministic)
        proposed = normalized.cpu().numpy()[0] * np.asarray(self.cfg.action_scale)
        limits = np.asarray(self.cfg.residual_scale) * np.asarray(self.cfg.action_scale)
        if not np.isfinite(proposed).all() or np.any(np.abs(proposed - base) > limits + 1e-6):
            raise RuntimeError("residual actor violated its action envelope")
        return proposed.astype(np.float32)

    def discard_observation(self):
        """Reject one unexecuted frame while retaining the active trial."""
        self.prior.discard_observation()
        self.current_observation = None
        self.base_action = None

    def capture(self, action, *, expert, command_sent):
        if self.records is None:
            return
        if self.current_observation is None:
            raise RuntimeError("missing observation for HIL transition")
        if len(self.records) >= self.cfg.replay_capacity:
            raise RuntimeError("HIL trial exceeds replay capacity; stop and label shorter trials")
        self.records.append({"observation": self.current_observation.copy(),
                             "action": np.asarray(action, dtype=np.float32).copy(),
                             "expert": bool(expert), "command_sent": bool(command_sent)})

    def finish_trial(self, raw_episode):
        if raw_episode is not None and self.records:
            self.pending[str(Path(raw_episode).resolve())] = (self.records, self.version)
        self.records = None
        self.current_observation = None

    def label(self, raw_episode, outcome):
        key = str(Path(raw_episode).resolve())
        records, version = self.pending[key]
        data = records_to_episode(records, outcome, self.cfg)
        identity = hashlib.sha256(key.encode()).hexdigest()
        destination = self.run / "online" / f"episode_{identity}.npz"
        write_episode(destination, data, self.cfg, metadata={
            "kind": "online", "raw_episode": key, "outcome": outcome,
            "policy_version": version, "act_sha256": self.cfg.act_sha256,
        })
        del self.pending[key]
        print(f"[HIL] committed {len(data['state'])} transitions ({outcome}) -> learner", flush=True)

    def discard(self):
        self.records = None
        self.current_observation = None

    def close(self):
        self.discard()
        self.prior.close()
