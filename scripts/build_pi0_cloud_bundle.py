"""Create an independent, checksummed cloud bundle without secrets or native SDKs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd

DEFAULT_DATASET = Path("data/lerobot/piper_pick_cup_dual_camera_lf_180eps")
DEFAULT_RUN_PREFIX = "pi0_lora_lf180"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def copy_stable(source, target):
    # Dereference whitelisted tokenizer pointers; never keep links back to PC.
    source = source.resolve(strict=True)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while packaging: {source}")


def copy_tree(source, target):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".cache", ".git"} or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file() and path.suffix != ".pyc":
            copy_stable(path, target / relative)


def compress_bundle(bundle, archive):
    """Archive only manifest files, excluding caches created by smoke checks."""
    partial = Path(str(archive) + ".partial")
    if archive.exists() or partial.exists():
        raise ValueError("Archive target already exists")
    manifest = json.loads((bundle / "CLOUD_MANIFEST.json").read_text())
    names = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        names.append(f"{bundle.name}/{relative.as_posix()}")
    names.append(f"{bundle.name}/CLOUD_MANIFEST.json")
    compressor = "pigz -p 8 -1" if shutil.which("pigz") else "gzip -1"
    subprocess.run(["tar", "-I", compressor, "-cf", str(partial), "-C", str(bundle.parent),
                    "--no-recursion", "--null", "--verbatim-files-from", "-T", "-"],
                   input=("\0".join(names) + "\0").encode(), check=True)
    partial.rename(archive)
    return archive


def render_launchers(bundle, dataset_relative, run_prefix, batch_size):
    old_dataset = DEFAULT_DATASET.as_posix()
    for name in ("run_smoke.sh", "run_train.sh", "check_smoke.sh"):
        path = bundle / "scripts/cloud" / name
        content = path.read_text(encoding="utf-8")
        if old_dataset not in content or DEFAULT_RUN_PREFIX not in content:
            raise ValueError(f"Unexpected cloud launcher template: {name}")
        content = content.replace(old_dataset, dataset_relative.as_posix())
        content = content.replace(DEFAULT_RUN_PREFIX, run_prefix)
        if name != "check_smoke.sh":
            if content.count("--batch-size 1") != 1:
                raise ValueError(f"Unexpected batch-size template: {name}")
            content = content.replace("--batch-size 1", f"--batch-size {batch_size}")
        path.write_text(content, encoding="utf-8")


def build(workspace, bundle, tokenizer_repo, dataset_relative=DEFAULT_DATASET, run_prefix=None,
          batch_size=1):
    workspace = workspace.resolve()
    bundle = bundle.resolve()
    dataset_relative = Path(dataset_relative)
    if dataset_relative.is_absolute():
        dataset_relative = dataset_relative.resolve().relative_to(workspace)
    if (dataset_relative.parts[:2] != ("data", "lerobot")
            or ".." in dataset_relative.parts or len(dataset_relative.parts) != 3):
        raise ValueError("Dataset must be one data/lerobot/<name> directory inside workspace")
    if run_prefix is None:
        run_prefix = (DEFAULT_RUN_PREFIX if dataset_relative == DEFAULT_DATASET
                      else "pi0_lora_" + dataset_relative.name)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_prefix):
        raise ValueError("Use a plain run prefix")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    archive = bundle.with_name(bundle.name + ".tar.gz")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle.name):
        raise ValueError("Use a plain bundle directory name")
    if (bundle.exists() or archive.exists() or Path(str(archive) + ".partial").exists()
            or Path(str(archive) + ".sha256").exists()):
        raise ValueError("Bundle/archive already exists; use a new destination")
    dataset = workspace / dataset_relative
    info_bytes = (dataset / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    task_frame = pd.read_parquet(dataset / "meta/tasks.parquet")
    if (task_frame.index.name != "task" or task_frame.index.has_duplicates
            or task_frame.columns.tolist() != ["task_index"]):
        raise ValueError(
            "Dataset tasks.parquet is not LeRobot-compatible: language task "
            "must be the pandas index named 'task'"
        )
    tasks = [
        {"task_index": int(row.task_index), "task": str(task)}
        for task, row in task_frame.sort_values("task_index").iterrows()
    ]
    if not tasks:
        raise ValueError("Dataset contains no language tasks")
    if [item["task_index"] for item in tasks] != list(range(len(tasks))):
        raise ValueError("Dataset task_index values must be contiguous from zero")
    ref = (tokenizer_repo / "refs/main").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("Invalid tokenizer revision")
    snapshot = tokenizer_repo / "snapshots" / ref
    token_files = ("config.json", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "added_tokens.json")
    for name in token_files:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    bundle.mkdir(parents=True)
    print("Copying code and dataset", flush=True)
    for name in ("pyproject.toml", "docs/README.md", "docs/BestPractice_and_LessonLearn.md"):
        copy_stable(workspace / name, bundle / name)
    copy_tree(workspace / "src/robotarm", bundle / "src/robotarm")
    copy_stable(workspace / "scripts/train_monitor.py", bundle / "scripts/train_monitor.py")
    copy_tree(workspace / "scripts/cloud", bundle / "scripts/cloud")
    render_launchers(bundle, dataset_relative, run_prefix, batch_size)
    copy_stable(workspace / "tests/test_pi0_lora.py", bundle / "tests/test_pi0_lora.py")
    trim_provenance = dataset / "meta/trim_provenance.json"
    origin = None
    if trim_provenance.is_file():
        provenance = json.loads(trim_provenance.read_text(encoding="utf-8"))
        origin = {"source_dataset": Path(provenance["source_dataset"]).name,
                  "source_episodes": len(provenance["reviews_snapshot"])}
    (bundle / "CLOUD_README.md").write_text(
        f"# π0 LoRA 云端训练包：{dataset_relative.name}\n\n"
        f"训练数据：`{dataset_relative}`；{info['total_episodes']} 条、"
        f"{info['total_frames']} 帧、{info['fps']} FPS。\n"
        + "语言任务：\n"
        + "".join(f"- `task_index={item['task_index']}`：`{item['task']}`\n" for item in tasks)
        + (f"来源：{origin['source_dataset']}（原始 {origin['source_episodes']} 条）；"
           "本包只包含已裁剪、已剔除的训练数据，不包含原始数据。\n" if origin else "")
        + "包含 π0 基础权重、tokenizer、训练代码与数据副本；不包含登录凭据、硬件 SDK 或训练结果。\n\n"
        "在压缩包所在目录校验并解压：\n\n"
        f"```bash\nsha256sum -c {bundle.name}.tar.gz.sha256\n"
        f"tar -xzf {bundle.name}.tar.gz\ncd {bundle.name}\n"
        "python3 scripts/cloud/verify_bundle.py\n```\n\n"
        "云端需要支持 BF16 的 CUDA GPU；首次安装依赖需要联网。使用 Python 3.12，"
        "并按云平台 CUDA 版本安装合适的 PyTorch：\n\n"
        "```bash\npython3.12 -m venv .venv-pi0\nsource .venv-pi0/bin/activate\n"
        "python -m pip install -e '.[pi-lora]'\npython -m pip check\n"
        "source scripts/cloud/cloud_env.sh\n```\n\n"
        "下列脚本默认只预览训练计划；确认后才加 `--execute`：\n\n"
        "```bash\nbash scripts/cloud/run_smoke.sh\n"
        "bash scripts/cloud/run_smoke.sh --execute\n"
        "bash scripts/cloud/check_smoke.sh\n"
        "bash scripts/cloud/run_train.sh\n"
        "bash scripts/cloud/run_train.sh --execute\n```\n\n"
        f"监控：`python scripts/train_monitor.py --run-dir artifacts/checkpoints/{run_prefix}_v1 --port 8420 --interval 2`。\n"
        f"试跑和正式训练的 batch size 均为 {batch_size}。默认正式配置："
        "rank=16、alpha=32、lr=1e-4、chunk=50、"
        "n_action_steps=5、eval_split=0.2、steps=20000。试跑为 20 步；"
        "先确认显存、loss 和重加载正常。重复训练不能覆盖旧输出目录。\n\n"
        "更多环境与迁回说明见 [README 云端流程](docs/README.md#85-云端包上传校验与启动)。\n",
        encoding="utf-8",
    )
    # Preserve figure assets referenced by README and the experiment ledger.
    copy_tree(workspace / "docs/best_practice_assets", bundle / "docs/best_practice_assets")
    copy_tree(dataset, bundle / dataset_relative)
    if (dataset / "meta/info.json").read_bytes() != info_bytes:
        raise RuntimeError("Dataset metadata changed while packaging; rebuild a stable snapshot")
    review_relative = Path("artifacts/reports/dataset_quality") / (dataset.name + ".json")
    source_review = workspace / review_relative
    if source_review.is_file():
        review = json.loads(source_review.read_text())
        previous_root = review.get("dataset_root")
        if previous_root and Path(previous_root).resolve() != dataset:
            raise ValueError("Review source belongs to another dataset")
        review["export_source_dataset_root"] = previous_root
        review["dataset_root"] = dataset_relative.as_posix()
        target = bundle / review_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Copying Pi0 base weights (independent copy)", flush=True)
    copy_tree(workspace / "artifacts/pretrained/pi0_base", bundle / "artifacts/pretrained/pi0_base")
    target_repo = bundle / ".cache/huggingface/hub" / tokenizer_repo.name
    for name in token_files:
        copy_stable(snapshot / name, target_repo / "snapshots" / ref / name)
    (target_repo / "refs").mkdir(parents=True)
    (target_repo / "refs/main").write_text(ref, encoding="utf-8")
    print("Hashing packaged files", flush=True)
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected packaged symlink: {path}")
        if path.is_file():
            files.append({"path": path.relative_to(bundle).as_posix(),
                          "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"format": "robotarm_pi0_cloud_v1",
                "dataset": {"path": dataset_relative.as_posix(), "episodes": info["total_episodes"],
                            "frames": info["total_frames"], "fps": info["fps"],
                            "tasks": tasks,
                            "origin": origin},
                "run_prefix": run_prefix,
                "batch_size": batch_size,
                "tokenizer_revision": ref, "files": files, "total_bytes": sum(f["bytes"] for f in files),
                "excluded": ["credentials", "raw sessions", "ACT checkpoints", "Conda environments", "native SDKs"]}
    (bundle / "CLOUD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Compressing {len(files)} files, {manifest['total_bytes'] / 2**30:.2f} GiB", flush=True)
    compress_bundle(bundle, archive)
    print("Hashing archive", flush=True)
    checksum = digest(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"DONE: {bundle}\nARCHIVE: {archive}\nSIZE: {archive.stat().st_size / 2**30:.2f} GiB", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET,
                        help="LeRobot dataset inside workspace; defaults to the historical LF180 dataset")
    parser.add_argument("--run-prefix", help="Training output prefix; default derives from dataset name")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for both smoke and formal training (default: 1)")
    args = parser.parse_args()
    build(args.workspace, args.bundle_dir, args.tokenizer_repo, args.dataset_root,
          args.run_prefix, args.batch_size)


if __name__ == "__main__":
    main()
