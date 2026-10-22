"""Create an independent, checksummed cloud bundle without secrets or native SDKs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


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


def build(workspace, bundle, tokenizer_repo):
    workspace = workspace.resolve()
    bundle = bundle.resolve()
    archive = bundle.with_name(bundle.name + ".tar.gz")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", bundle.name):
        raise ValueError("Use a plain bundle directory name")
    if (bundle.exists() or archive.exists() or Path(str(archive) + ".partial").exists()
            or Path(str(archive) + ".sha256").exists()):
        raise ValueError("Bundle/archive already exists; use a new destination")
    dataset_relative = Path("data/lerobot/piper_pick_cup_dual_camera_lf_180eps")
    dataset = workspace / dataset_relative
    info_bytes = (dataset / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
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
    copy_stable(workspace / "tests/test_pi0_lora.py", bundle / "tests/test_pi0_lora.py")
    # Keep the cloud entry point, but maintain operational instructions only in README.
    (bundle / "CLOUD_README.md").write_text(
        "# π0 LoRA 云端训练包\n\n"
        "上传、校验、环境安装、试跑与正式训练见 "
        "[README 云端流程](docs/README.md#85-云端包上传校验与启动)。\n",
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
                            "frames": info["total_frames"], "fps": info["fps"]},
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
    args = parser.parse_args()
    build(args.workspace, args.bundle_dir, args.tokenizer_repo)


if __name__ == "__main__":
    main()
