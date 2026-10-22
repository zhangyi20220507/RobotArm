"""工作流默认参数的集中管理（YAML 单点真理）。

默认值不再散落在各模块的 argparse 里：本目录按 CLI 组名组织 YAML 文件
（data_acquisition.yaml / modeling.yaml），文件内以工作流名为键存放
该工作流的全部默认参数。argparse 仅保留 schema（flag、类型、help）。

优先级：命令行显式参数 > YAML 默认值。
路径类默认（数据根、报告、审批文件等）仍由 `robotarm.paths` 统一管理，
不在此处重复定义。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_ROOT = Path(
    os.environ.get("ROBOTARM_CONFIG_DIR", Path(__file__).resolve().parent)
)


@lru_cache(maxsize=None)
def _load_group(group: str) -> dict:
    path = CONFIG_ROOT / f"{group}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing workflow config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid workflow config (not a mapping): {path}")
    return data


def defaults(group: str, workflow: str) -> dict:
    """读取 <组>/<工作流> 的默认参数（返回副本）。

    group: CLI 组名（data_acquisition / modeling）
    workflow: 工作流名（与 CLI 一致，如 collect-piper）
    """
    section = _load_group(group).get(workflow)
    if not isinstance(section, dict):
        raise KeyError(f"config section missing: {group}.yaml -> {workflow}")
    return dict(section)


def policy_defaults(group: str, workflow: str, policy_type: str) -> dict:
    """读取工作流默认参数，并叠加按策略类型的子节（policies.<policy_type>）。

    用于同一工作流对不同 policy（act/diffusion/smolvla）需要不同默认值的场景
    （如推理时限）。键名与 argparse dest 一致（下划线）。
    """
    merged = defaults(group, workflow)
    overlays = merged.pop("policies", {})
    if not isinstance(overlays, dict):
        raise ValueError(
            f"invalid policies section (not a mapping): {group}.yaml -> {workflow}"
        )
    overlay = overlays.get(policy_type, {})
    if not isinstance(overlay, dict):
        raise ValueError(
            f"invalid policy overlay: {group}.yaml -> {workflow}.policies.{policy_type}"
        )
    merged.update(overlay)
    return merged
