"""Serializable contracts shared by preparation, learner and physical actor."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


def checkpoint_hash(path):
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        file = Path(path) / name
        digest.update(name.encode())
        digest.update(file.read_bytes())
    for file in sorted(Path(path).glob("*processor*.safetensors")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass
class Config:
    act_checkpoint: str
    act_sha256: str
    observation_dim: int
    camera_roles: list[str]
    format_version: str = "piper_hil_residual_sac_v1"
    task: str = ""
    camera_serials: dict[str, str] = field(default_factory=dict)
    rate_hz: float = 20.0
    pool_size: int = 2
    hidden_dim: int = 256
    max_joint_delta_deg: float = 1.0
    max_gripper_delta_mm: float = 2.0
    gripper_max_m: float = 0.1
    discount: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.0003
    initial_alpha: float = 0.01
    bc_weight: float = 0.1
    batch_size: int = 128
    replay_capacity: int = 100000
    seed: int = 1000
    temporal_ensemble_coeff: float | None = None

    def validate(self):
        if self.format_version not in {"piper_hil_residual_sac_v1", "piper_hil_residual_sac_v2"}:
            raise ValueError("unsupported HIL run format")
        if self.format_version.endswith("v2"):
            if (self.temporal_ensemble_coeff is None
                    or not math.isfinite(self.temporal_ensemble_coeff)
                    or self.temporal_ensemble_coeff <= 0):
                raise ValueError("HIL v2 requires a positive ACT temporal ensemble coefficient")
        elif self.temporal_ensemble_coeff is not None:
            raise ValueError("legacy HIL caches use a stateless ACT reference")
        if self.observation_dim < 14 or self.hidden_dim < 8 or self.pool_size < 1:
            raise ValueError("invalid HIL network dimensions")
        if not self.camera_roles or len(set(self.camera_roles)) != len(self.camera_roles):
            raise ValueError("camera roles must be non-empty and unique")
        if self.batch_size < 2 or self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must cover a batch of at least two")
        if not math.isfinite(self.rate_hz) or not 1 <= self.rate_hz <= 20:
            raise ValueError("Piper HIL rate must be within 1..20 Hz")
        for value in (self.max_joint_delta_deg, self.max_gripper_delta_mm,
                      self.gripper_max_m, self.learning_rate, self.initial_alpha):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("HIL limits and learning rates must be finite and positive")
        if not 0 < self.discount < 1 or not 0 < self.tau <= 1 or not math.isfinite(self.bc_weight) or self.bc_weight < 0:
            raise ValueError("invalid SAC discount, target update or BC weight")
        if self.max_joint_delta_deg > 5 or self.max_gripper_delta_mm > 10:
            raise ValueError("residual limits exceed supported 5 deg / 10 mm envelope")
        return self

    @property
    def action_scale(self):
        return [math.pi] * 6 + [self.gripper_max_m]

    @property
    def residual_scale(self):
        return [math.radians(self.max_joint_delta_deg) / math.pi] * 6 + [
            self.max_gripper_delta_mm / 1000 / self.gripper_max_m]

    def save(self, path):
        self.validate()
        atomic_json(path, asdict(self))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    @property
    def signature(self):
        contract = asdict(self)
        if self.format_version.endswith("v1"):
            contract.pop("temporal_ensemble_coeff")  # Preserve old cache/weight signatures.
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
