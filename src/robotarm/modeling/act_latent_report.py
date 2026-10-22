"""Offline ACT A/B/C/D latent comparison. No robot or camera connection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import statistics


@contextmanager
def latent_mode(policy, reconstruction: bool):
    """Enable ACT's posterior branch only, leaving all dropout/BN in eval.

    In the installed LeRobot ACT, only the outer ACT.training flag selects
    posterior sampling. Calling train() recursively here would confound A/C.
    Restore every flag even if forward fails.
    """
    flags = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        policy.model.training = reconstruction
        yield
    finally:
        for module, training in flags:
            module.training = training


def predict(policy, batch, reconstruction):
    from lerobot.utils.constants import OBS_IMAGES

    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = [model_batch[key] for key in policy.config.image_features]
    with latent_mode(policy, reconstruction):
        prediction, (mu, logvar) = policy.model(model_batch)
    if reconstruction != (mu is not None and logvar is not None):
        raise RuntimeError("ACT latent branch mismatch; installed LeRobot API may have changed")
    return prediction


def l1_totals(prediction, target, is_pad):
    """Sum per action dimension and count valid timesteps (not padded slots)."""
    import torch

    if prediction.shape != target.shape or is_pad.shape != target.shape[:2]:
        raise ValueError("Expected matching [batch,chunk,action] and [batch,chunk] padding")
    valid = ~is_pad.bool()
    errors = (prediction - target).abs()[valid]
    if not torch.isfinite(errors).all():
        raise ValueError("Non-finite error at a valid action position")
    return errors.double().sum(dim=0).cpu(), int(valid.sum().item())


def sample_indices(length, limit):
    count = min(length, limit) if limit else length
    if count <= 0:
        raise ValueError("Dataset split has no frames")
    return [i * length // count for i in range(count)]


def summarize(sums, count):
    if not count:
        raise ValueError("No non-padding actions")
    per_repeat = [(total / count).tolist() for total in sums]
    losses = [statistics.mean(values) for values in per_repeat]
    return {
        "l1": statistics.mean(losses),
        "repeat_std": statistics.pstdev(losses),
        "repeat_l1": losses,
        "per_dimension_l1": [statistics.mean(col) for col in zip(*per_repeat)],
        "valid_timesteps_per_repeat": count,
        "valid_elements_per_repeat": count * len(per_repeat[0]),
    }


def evaluate_split(policy, preprocessor, dataset, args):
    import torch
    from torch.utils.data import DataLoader, Subset

    indices = sample_indices(len(dataset), args.max_frames)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    reconstruction_sums = [None] * args.repeats
    zero_sum = None
    valid_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            for key in dataset.meta.camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            batch = preprocessor(batch)
            target, padding = batch["action"], batch["action_is_pad"]
            total, count = l1_totals(predict(policy, batch, False), target, padding)
            zero_sum = total if zero_sum is None else zero_sum + total
            valid_count += count
            for repeat in range(args.repeats):
                # Explicit per-batch seeds: deterministic for a fixed batch
                # size and sample manifest, independent of the zero-z pass.
                seed = args.seed + repeat * 1_000_003 + batch_index
                devices = [torch.cuda.current_device()] if args.device == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    total, _ = l1_totals(predict(policy, batch, True), target, padding)
                reconstruction_sums[repeat] = total if reconstruction_sums[repeat] is None else reconstruction_sums[repeat] + total
            print(f"frames {min((batch_index + 1) * args.batch_size, len(indices))}/{len(indices)}", flush=True)
    return {
        "episodes": list(dataset.episodes), "total_frames": len(dataset),
        "evaluated_frames": len(indices), "sample_indices": indices,
        "reconstruction": summarize(reconstruction_sums, valid_count),
        "zero": summarize([zero_sum], valid_count),
    }


def render_report(report):
    escape = html.escape
    train, val = report["train"], report["validation"]
    cells = [("A", train["reconstruction"]), ("B", train["zero"]),
             ("C", val["reconstruction"]), ("D", val["zero"])]
    def cell(label, metric):
        return f'<td><strong>{label}: {metric["l1"]:.6f}</strong><br>重复标准差 {metric["repeat_std"]:.6f}</td>'
    rows = "".join(f'<tr><th>{name}</th>{cell(*cells[i])}{cell(*cells[i+1])}</tr>'
                   for name, i in (("Train", 0), ("Validation", 2)))
    details = ""
    for name, split in (("Train", train), ("Validation", val)):
        details += f'<h3>{name}</h3><p>评估 {split["evaluated_frames"]} / {split["total_frames"]} 帧；有效动作元素/次：{split["zero"]["valid_elements_per_repeat"]}</p>'
        details += f'<details><summary>Episode 划分</summary><pre>{escape(str(split["episodes"]))}</pre></details>'
    dim_rows = "".join('<tr><th>' + str(i) + '</th>' + ''.join(
        f'<td>{metric["per_dimension_l1"][i]:.6f}</td>' for _, metric in cells) + '</tr>'
        for i in range(len(cells[0][1]["per_dimension_l1"])))
    a, b, c, d = [metric["l1"] for _, metric in cells]
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>ACT 潜变量 L1 对照</title><style>body{{font:16px system-ui;max-width:1050px;margin:36px auto;padding:0 20px;color:#203040}}table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{border:1px solid #ccd5df;padding:14px;text-align:left}}th{{background:#eef3f8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.note{{background:#fff3cf;padding:16px}}</style>
<h1>ACT 潜变量 L1 对照</h1><p>Checkpoint：{escape(report["checkpoint"])}</p>
<p class="note">{"抽样报告：不是完整数据集结果。" if report["sampled"] else "完整训练/验证划分评估。"}
四组均关闭 dropout、数据增强和 BatchNorm 更新。A/C 仅开启真实动作的 VAE 编码及随机潜变量采样，不是带 dropout 的训练日志复算。所有指标只含 L1，不含 KL。</p>
<table><tr><th>数据集</th><th>真实动作编码潜变量</th><th>z = 0</th></tr>{rows}</table>
<p>训练集潜变量差距 B−A：{b-a:.6f}；验证集潜变量差距 D−C：{d-c:.6f}；推理式泛化差距 D−B：{d-b:.6f}；重建差距 C−A：{c-a:.6f}。</p>
<p>较大的 B−A / D−C 提示模型对真实动作编码的潜变量有依赖；D−B 反映此划分上的推理式差距。这些差值不能单独证明过拟合原因，也不等同于真机成功率。</p>
<h2>统计口径</h2><p>在 checkpoint 预处理后的动作空间计算整段 chunk L1；排除 action_is_pad；总绝对误差除以总有效动作元素数，而非 batch 均值的简单平均。重叠 chunk 的未来动作按各预测位置计数。无动作输出后处理，无 online-policy 限制。</p>
<p>重建重复 {report["repeats"]} 次；seed={report["seed"]}；每批 seed = seed + repeat×1000003 + batch_index。固定样本及 batch size 才能复现同一随机序列。z=0 仅计算一次。FP32，无 autocast。标准差为重复间总体标准差，不是置信区间。</p>
{details}<h2>各动作维度 L1（归一化空间，非度/mm）</h2><table><tr><th>维度（从0起）</th><th>A</th><th>B</th><th>C</th><th>D</th></tr>{dim_rows}</table>
<details><summary>完整指标与复现信息</summary><pre>{escape(json.dumps(report,ensure_ascii=False,indent=2))}</pre></details></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="Override location only; preserve saved episode split")
    parser.add_argument("--output", type=Path, required=True, help="HTML path; JSON written alongside")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0, help="Per split; 0=all, otherwise evenly sampled")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.repeats, args.cpu_threads) < 1 or args.max_frames < 0 or args.seed < 0:
        parser.error("Positive batch/repeats/threads and nonnegative max-frames/seed required")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from ..paths import WORKSPACE_ROOT
    os.environ.setdefault("HF_DATASETS_CACHE", str(WORKSPACE_ROOT / ".cache/huggingface/datasets"))
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for offline evaluation")
    checkpoint = args.checkpoint.resolve()
    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    if cfg.policy.type != "act" or not cfg.policy.use_vae:
        raise ValueError("Requires ACT checkpoint with use_vae=true")
    if cfg.dataset.eval_split <= 0 or cfg.dataset.streaming:
        raise ValueError("Requires saved non-streaming train/validation split")
    if args.dataset_root:
        cfg.dataset.root = str(args.dataset_root.resolve())
    cfg.dataset.image_transforms.enable = False
    cfg.policy.device = args.device
    cfg.policy.pretrained_path = str(checkpoint)
    train, validation = make_train_eval_datasets(cfg)
    if validation is None or set(train.episodes) & set(validation.episodes):
        raise ValueError("Missing or overlapping validation episodes")
    policy = make_policy(cfg.policy, ds_meta=train.meta).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    report = {"checkpoint": str(checkpoint), "dataset": str(cfg.dataset.root),
              "train_config_sha256": hashlib.sha256((checkpoint / "train_config.json").read_bytes()).hexdigest(),
              "seed": args.seed, "repeats": args.repeats, "batch_size": args.batch_size,
              "device": args.device, "torch_version": torch.__version__, "chunk_size": cfg.policy.chunk_size}
    for name, dataset in (("train", train), ("validation", validation)):
        print(f"Evaluating {name}: {len(dataset.episodes)} episodes", flush=True)
        report[name] = evaluate_split(policy, preprocessor, dataset, args)
    report["sampled"] = any(report[name]["evaluated_frames"] < report[name]["total_frames"] for name in ("train", "validation"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
