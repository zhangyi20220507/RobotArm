"""Generate a local Markdown/CSV/PNG report from a LeRobot WandB run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_history(run_file: Path) -> list[dict[str, float]]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(run_file))
    rows = []
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.history.item:
            continue
        row = {}
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                value = json.loads(item.value_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
        if row:
            rows.append(row)
    return rows


def read_run_history(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Merge all resume segments; deduplicate copied/latest-run WandB files.

    At an overlapping step the newer timestamp wins for repeated keys. Separate
    train/eval records at the same step are combined. Inputs are never modified.
    """
    merged, sources, seen = {}, [], set()
    for path in sorted(run_dir.glob("wandb/**/run-*.wandb")):
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows = [r for r in read_history(path) if "_step" in r and math.isfinite(r["_step"])]
        if not rows:
            continue
        sources.append({"path": str(path.resolve()), "sha256": digest,
                        "first_step": int(min(r["_step"] for r in rows)),
                        "last_step": int(max(r["_step"] for r in rows))})
        for row in rows:
            step = int(row["_step"])
            previous = merged.get(step, {})
            if row.get("_timestamp", 0) >= previous.get("_timestamp", 0):
                merged[step] = {**previous, **row}
            else:
                merged[step] = {**row, **previous}
    if not merged:
        raise ValueError(f"no history records found under {run_dir}")
    return [merged[s] for s in sorted(merged)], sources


def summarize_history(rows: list[dict], run_dir: Path) -> dict:
    """Select the actual best evaluated, saved checkpoint, never the nearest."""
    rows = sorted(rows, key=lambda r: r["_step"])
    eval_points = [(int(r["_step"]), r["eval/eval_loss"]) for r in rows
                   if "eval/eval_loss" in r and math.isfinite(r["eval/eval_loss"])]
    if not eval_points:
        raise ValueError("no finite validation loss in run history")
    saved = {int(p.parent.parent.name) for p in (run_dir / "checkpoints").glob("*/pretrained_model/config.json")
             if p.parent.parent.name.isdigit()}
    best_step, best_eval = min(eval_points, key=lambda x: x[1])
    candidates = [(s, v) for s, v in eval_points if s in saved]
    best_saved = min(candidates, key=lambda x: x[1]) if candidates else None
    def last_value(key):
        return next((r[key] for r in reversed(rows) if key in r), None)
    return {"run_dir": str(run_dir), "final_step": int(rows[-1]["_step"]),
            "final_train_loss": last_value("train/loss"),
            "final_train_l1_loss": last_value("train/l1_loss"),
            "final_eval_loss": eval_points[-1][1], "final_eval_step": eval_points[-1][0],
            "best_eval_step": best_step, "best_eval_loss": best_eval,
            "recommended_checkpoint_step": best_saved[0] if best_saved else None,
            "recommended_checkpoint_eval_loss": best_saved[1] if best_saved else None,
            "recommended_checkpoint": str(run_dir / "checkpoints" / f"{best_saved[0]:06d}" / "pretrained_model") if best_saved else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    rows, sources = read_run_history(run_dir)
    summary = summarize_history(rows, run_dir)
    summary["history_sources"] = sources
    output = (args.output_dir or run_dir / "training_report").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    keys = ["_step", "train/loss", "train/l1_loss", "train/kld_loss", "eval/eval_loss"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for key, label in (("train/loss", "total"), ("train/l1_loss", "L1"), ("train/kld_loss", "KLD")):
        points = [(r.get("_step"), r.get(key)) for r in rows if key in r and "_step" in r]
        if points:
            axes[0].plot(*zip(*points), label=label, linewidth=1.4)
    axes[0].set_ylabel("Training loss")
    axes[0].grid(alpha=.25)
    axes[0].legend()
    eval_points = [(r["_step"], r["eval/eval_loss"]) for r in rows if "eval/eval_loss" in r]
    axes[1].plot(*zip(*eval_points), color="#e76f51", label="holdout eval loss", linewidth=1.8)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Evaluation loss")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(output / "loss_curves.png", dpi=160)
    plt.close(fig)

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    def number(value):
        return f"{value:.6f}" if value is not None else "未记录"
    report = f"""# {run_dir.name} 训练报告

![Loss curves](loss_curves.png)

| 指标 | 数值 |
|---|---:|
| 最终 step | {summary['final_step']} |
| 最终 train loss | {number(summary['final_train_loss'])} |
| 最终 train L1 loss | {number(summary['final_train_l1_loss'])} |
| 最后 eval loss | {summary['final_eval_loss']:.6f}（step {summary['final_eval_step']}） |
| 最低 eval loss | {summary['best_eval_loss']:.6f}（step {summary['best_eval_step']}） |
| 推荐 checkpoint | `{summary['recommended_checkpoint']}` |
| 推荐项的实际 eval loss | {number(summary['recommended_checkpoint_eval_loss'])} |

> 合并全部续训日志、去除重复副本。只在有对应验证值的已保存 checkpoint 中选最低项；
> 最低日志点未保存时，不用附近权重冒充。无候选时不推荐。最终选择仍需验证闭环任务效果。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
