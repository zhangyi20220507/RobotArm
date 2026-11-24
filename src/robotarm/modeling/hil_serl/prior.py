"""Frozen ACT inference and spatial visual features, shared offline and online."""
from pathlib import Path
import copy

import numpy as np
import torch
from torch.nn import functional as F


class ACTPrior:
    def __init__(self, policy, preprocessor, postprocessor, *, pool_size=2, gripper_max_m=0.1,
                 temporal_ensemble_coeff=None):
        self.policy = policy.eval()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.pool_size = pool_size
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self._previous_ensemble = None
        if temporal_ensemble_coeff is not None:
            if (policy.config.n_action_steps != 1
                    or policy.config.temporal_ensemble_coeff != temporal_ensemble_coeff):
                raise ValueError("live ACT temporal ensemble differs from the HIL cache contract")
        self.scale = np.array([np.pi] * 6 + [gripper_max_m], dtype=np.float32)
        self.roles = [key.removeprefix("observation.images.") for key in policy.config.image_features]
        self._visual = []
        self._hook = policy.model.backbone.register_forward_hook(self._capture)
        for p in policy.parameters():
            p.requires_grad_(False)

    def _capture(self, module, inputs, output):
        self._visual.append(F.adaptive_avg_pool2d(output["feature_map"], self.pool_size).flatten(1))

    @classmethod
    def load(cls, checkpoint, *, device="cpu", pool_size=2, gripper_max_m=0.1,
             temporal_ensemble_coeff=None):
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = ACTConfig.from_pretrained(str(checkpoint))
        config.device = device
        config.pretrained_backbone_weights = None  # All backbone weights are in the ACT checkpoint.
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy = ACTPolicy.from_pretrained(str(checkpoint), config=config).to(device)
        pre, post = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return cls(policy, pre, post, pool_size=pool_size, gripper_max_m=gripper_max_m,
                   temporal_ensemble_coeff=temporal_ensemble_coeff)

    def reset(self):
        self.policy.reset()
        self._previous_ensemble = None

    def discard_observation(self):
        if self._previous_ensemble is not None:
            self.policy.temporal_ensembler = self._previous_ensemble
            self._previous_ensemble = None

    @torch.inference_mode()
    def encode(self, batch, *, reference_action=None):
        observations, actions = self.encode_batch(batch, reference_action=reference_action)
        if len(observations) != 1:
            raise ValueError("live ACT encoding requires exactly one observation")
        return observations[0], actions[0]

    @torch.inference_mode()
    def encode_batch(self, batch, *, reference_action=None):
        self._visual.clear()
        self._previous_ensemble = None
        prepared = self.preprocessor(dict(batch))
        if self.temporal_ensemble_coeff is not None and reference_action is None:
            if batch["observation.state"].shape[0] != 1:
                raise ValueError("temporal ACT references must be encoded sequentially")
            self._previous_ensemble = copy.deepcopy(self.policy.temporal_ensembler)
            normalized = self.policy.select_action(prepared)
        else:
            # Legacy caches use a stateless reference. During a live safety
            # hold, capture fresh visual features without advancing ACT's clock.
            normalized = self.policy.predict_action_chunk(prepared)[:, 0]
        action = self.postprocessor(normalized).detach().cpu().numpy().astype(np.float32)
        if reference_action is not None:
            action = np.asarray(reference_action, dtype=np.float32).reshape(1, 7)
        if len(self._visual) != len(self.roles):
            raise RuntimeError("ACT backbone did not emit one feature map per configured camera")
        visual = torch.cat(self._visual, -1).float().cpu().numpy()
        state = batch["observation.state"].detach().cpu().numpy()
        observation = np.concatenate((visual, state / self.scale, action / self.scale), axis=-1).astype(np.float32)
        if state.ndim != 2 or state.shape[1] != 7 or action.shape != state.shape or not np.isfinite(observation).all():
            raise ValueError("invalid ACT state, action or visual features")
        return observation, action

    def close(self):
        self._hook.remove()


def dataset_batch(sample, roles):
    batch = {"observation.state": sample["observation.state"].float().reshape(1, 7)}
    for role in roles:
        key = "observation.images." + role
        image = sample[key]
        if image.dtype == torch.uint8:
            image = image.float() / 255
        batch[key] = image.float().unsqueeze(0)
    return batch
