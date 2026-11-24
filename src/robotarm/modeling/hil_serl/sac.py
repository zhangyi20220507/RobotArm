"""Twin-Q SAC in absolute action coordinates with a bounded ACT residual actor.

Replay stores the actual executed action, including human corrections outside the
residual envelope. Actor entropy is measured in normalized residual coordinates.
The frozen visual/ACT observation ends with the normalized ACT baseline (7D).
"""
import copy
import math
import os
from pathlib import Path
import uuid

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ResidualActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mlp(cfg.observation_dim, cfg.hidden_dim, 14)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("limit", torch.tensor(cfg.residual_scale))

    def forward(self, observation, deterministic=False):
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 1)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else distribution.rsample()
        residual = z.tanh()
        # Stable tanh Jacobian correction; entropy tuning excludes fixed unit scales.
        log_prob = (distribution.log_prob(z) - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1, keepdim=True)
        action = observation[..., -7:] + residual * self.limit
        action = torch.cat((action[..., :6], action[..., 6:].clamp(0, 1)), dim=-1)
        return action, log_prob


class SAC:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg.validate()
        self.device = torch.device(device)
        self.actor = ResidualActor(cfg).to(self.device)
        self.q1 = mlp(cfg.observation_dim + 7, cfg.hidden_dim, 1).to(self.device)
        self.q2 = mlp(cfg.observation_dim + 7, cfg.hidden_dim, 1).to(self.device)
        self.t1, self.t2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.t1.parameters()) + list(self.t2.parameters()):
            p.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha), device=self.device))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.learning_rate)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
        self.step = 0

    def update(self, batch):
        b = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
        state, next_state, action = b["state"], b["next_state"], b["action"]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_logp = self.actor(next_state)
            next_input = torch.cat((next_state, next_action), -1)
            target = b["reward"] + self.cfg.discount * (1 - b["done"]) * (
                torch.minimum(self.t1(next_input), self.t2(next_input)) - alpha * next_logp)
        actual = torch.cat((state, action), -1)
        critic_loss = F.mse_loss(self.q1(actual), target) + F.mse_loss(self.q2(actual), target)
        self._optimize(critic_loss, self.q_opt, list(self.q1.parameters()) + list(self.q2.parameters()))
        critics = list(self.q1.parameters()) + list(self.q2.parameters())
        for p in critics:
            p.requires_grad_(False)
        try:
            sampled, logp = self.actor(state)
            predicted_q = torch.minimum(self.q1(torch.cat((state, sampled), -1)), self.q2(torch.cat((state, sampled), -1)))
            actor_loss = (alpha * logp - predicted_q).mean()
            # Expert regularization targets only corrections reachable by the residual actor.
            mean, _ = self.actor(state, deterministic=True)
            desired = torch.maximum(torch.minimum(action, state[:, -7:] + self.actor.limit), state[:, -7:] - self.actor.limit)
            bc = (((mean - desired) / self.actor.limit).square().mean(-1, keepdim=True) * b["expert"]).sum() / b["expert"].sum().clamp_min(1)
            actor_loss = actor_loss + self.cfg.bc_weight * bc
            self._optimize(actor_loss, self.actor_opt, list(self.actor.parameters()))
        finally:
            for p in critics:
                p.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (logp.detach() - 7)).mean()
        self._optimize(alpha_loss, self.alpha_opt, [self.log_alpha])
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(1e-6), math.log(1))
            for target_net, online in ((self.t1, self.q1), (self.t2, self.q2)):
                for target_p, p in zip(target_net.parameters(), online.parameters()):
                    target_p.lerp_(p, self.cfg.tau)
        self.step += 1
        return {"step": self.step, "critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach()), "alpha": float(alpha), "bc_loss": float(bc.detach())}

    @staticmethod
    def _optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SAC loss; refusing to publish weights")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()
        return norm

    def save(self, path, *, training=False, extra=None):
        payload = {"signature": self.cfg.signature, "step": self.step,
                   "actor": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()}}
        if training:
            payload.update({"q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                            "t1": self.t1.state_dict(), "t2": self.t2.state_dict(),
                            "log_alpha": self.log_alpha.detach(),
                            "actor_opt": self.actor_opt.state_dict(), "q_opt": self.q_opt.state_dict(),
                            "alpha_opt": self.alpha_opt.state_dict(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            "extra": extra or {}})
        atomic_torch_save(path, payload)

    def restore(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        if data["signature"] != self.cfg.signature:
            raise ValueError("SAC checkpoint belongs to another run / ACT prior")
        for key in ("actor", "q1", "q2", "t1", "t2"):
            getattr(self, key).load_state_dict(data[key])
        for key in ("actor_opt", "q_opt", "alpha_opt"):
            getattr(self, key).load_state_dict(data[key])
        with torch.no_grad():
            self.log_alpha.copy_(data["log_alpha"])
        self.step = int(data["step"])
        torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in data["cuda_rng"]])
        return data.get("extra", {})


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_actor(path, cfg, device="cpu", minimum_step=1):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["signature"] != cfg.signature or int(payload["step"]) < minimum_step:
        raise ValueError("actor snapshot is incompatible or has not completed training")
    actor = ResidualActor(cfg).to(device)
    actor.load_state_dict(payload["actor"])
    if any(not torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError("actor snapshot contains non-finite parameters")
    actor.eval()
    return actor, int(payload["step"])
