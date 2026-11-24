#!/usr/bin/env python3
"""Verify the repository-managed LeRobot ACT layer3/layer4 extension."""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import fields

import torch


SUPPORTED_LEROBOT_VERSION = "0.6.1"


def _make_config(feature_layer: str):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    return ACTConfig(
        input_features={
            "observation.images.test": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        device="cpu",
        pretrained_backbone_weights=None,
        vision_feature_layer=feature_layer,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
    )


def _verify_model(feature_layer: str, expected_channels: int, expected_hw: tuple[int, int]) -> None:
    from lerobot.policies.act.modeling_act import ACT
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    model = ACT(_make_config(feature_layer)).eval()
    with torch.inference_mode():
        feature_map = model.backbone(torch.zeros(1, 3, 64, 64))["feature_map"]
        actions, _ = model(
            {
                OBS_IMAGES: [torch.zeros(1, 3, 64, 64)],
                OBS_STATE: torch.zeros(1, 7),
            }
        )

    actual_shape = tuple(feature_map.shape)
    expected_shape = (1, expected_channels, *expected_hw)
    if actual_shape != expected_shape:
        raise RuntimeError(f"{feature_layer}: expected feature shape {expected_shape}, got {actual_shape}")
    if model.encoder_img_feat_input_proj.in_channels != expected_channels:
        raise RuntimeError(
            f"{feature_layer}: projection expects {model.encoder_img_feat_input_proj.in_channels} channels, "
            f"expected {expected_channels}"
        )
    if tuple(actions.shape) != (1, 2, 7):
        raise RuntimeError(f"{feature_layer}: expected action shape (1, 2, 7), got {tuple(actions.shape)}")
    print(
        f"OK {feature_layer}: feature_map={actual_shape}, "
        f"projection_in={model.encoder_img_feat_input_proj.in_channels}, actions={tuple(actions.shape)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-version",
        default=SUPPORTED_LEROBOT_VERSION,
        help="Exact LeRobot version expected by this repository patch.",
    )
    args = parser.parse_args()

    installed_version = importlib.metadata.version("lerobot")
    if installed_version != args.allow_version:
        raise RuntimeError(
            f"This patch supports lerobot=={args.allow_version}; found lerobot=={installed_version}."
        )

    from lerobot.policies.act.configuration_act import ACTConfig

    field_names = {item.name for item in fields(ACTConfig)}
    if "vision_feature_layer" not in field_names:
        raise RuntimeError("ACTConfig has no vision_feature_layer field; apply the repository patch first.")
    if ACTConfig.__dataclass_fields__["vision_feature_layer"].default != "layer4":
        raise RuntimeError("vision_feature_layer must default to layer4 for old-checkpoint compatibility.")

    _verify_model("layer3", expected_channels=256, expected_hw=(4, 4))
    _verify_model("layer4", expected_channels=512, expected_hw=(2, 2))
    print(f"OK lerobot=={installed_version}: ACT feature-layer extension is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
