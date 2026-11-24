"""Build separate code-only Pi0 and SmolVLA cloud training archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


POLICY_DEFAULTS = {
    "pi0": {
        "base": "artifacts/pretrained/pi0_base",
        "batch": 1,
        "run": "artifacts/checkpoints/pi0_task_270eps_lora_v1",
    },
    "smolvla": {
        "base": "artifacts/pretrained/smolvla_base",
        "batch": 4,
        "run": "artifacts/checkpoints/smolvla_task_270eps_lora_v1",
    },
}


def copy_tree(source: Path, target: Path):
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def render_job_env(policy: str) -> str:
    settings = POLICY_DEFAULTS[policy]
    training_mode = "expert" if policy == "smolvla" else "lora"
    return f"""# Generated package defaults. Paths are relative to the unpacked package root.
POLICY={policy}
TRAINING_MODE={training_mode}
DATASET_ROOT=data/lerobot/task_dataset
BASE_CHECKPOINT={settings['base']}
OUTPUT_DIR={settings['run']}

BATCH_SIZE={settings['batch']}
STEPS=30000
CHUNK_SIZE=50
N_ACTION_STEPS=20
EVAL_SPLIT=0.1
SAVE_FREQ=2000
EVAL_STEPS=2000
MAX_EVAL_SAMPLES=256
LORA_RANK=16
LORA_ALPHA=32
LEARNING_RATE=0.0001
NUM_WORKERS=4
SEED=1000
QUALITY_REVIEWS=
MONITOR_PORT=8420
RESUME_CONFIG=
"""


def copy_package_files(workspace: Path, package: Path, policy: str):
    required_files = (
        "pyproject.toml",
        "requirements/unified-constraints.txt",
        "scripts/train_monitor.py",
        "docs/CLOUD_VLA_TRAINING.md",
    )
    for relative in required_files:
        source = workspace / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copy_tree(workspace / "src/robotarm", package / "src/robotarm")
    copy_tree(workspace / "scripts/cloud_vla", package / "scripts/cloud_vla")
    shutil.copy2(workspace / "docs/CLOUD_VLA_TRAINING.md", package / "README.md")
    (package / "job.env").write_text(render_job_env(policy), encoding="utf-8")
    (package / "data/lerobot").mkdir(parents=True)
    (package / "artifacts/pretrained").mkdir(parents=True)
    (package / "artifacts/checkpoints").mkdir(parents=True)
    manifest = {
        "format": "robotarm_vla_cloud_code_v1",
        "policy": policy,
        "lerobot": "0.6.1",
        "torch": "2.11.0+cu128",
        "contains": ["training code", "environment installer", "launch config"],
        "excluded": ["dataset", "model weights", "checkpoints", "credentials", "hardware SDKs"],
    }
    (package / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def archive_package(package: Path, output: Path):
    with tarfile.open(output, "w:gz") as archive:
        archive.add(package, arcname=package.name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return checksum


def build(workspace: Path, output_dir: Path, prefix: str):
    workspace = workspace.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="robotarm-vla-bundles-") as temporary:
        staging = Path(temporary)
        for policy in ("pi0", "smolvla"):
            package = staging / f"{prefix}_{policy}"
            package.mkdir()
            copy_package_files(workspace, package, policy)
            archive = output_dir / f"{package.name}.tar.gz"
            checksum = archive_package(package, archive)
            results.append((archive, checksum))
    return results


def build_directories(workspace: Path, output_dir: Path, prefix: str):
    """Materialize editable, code-only package directories without archiving them."""
    workspace = workspace.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for policy in ("pi0", "smolvla"):
        package = output_dir / f"{prefix}_{policy}"
        if package.exists():
            raise FileExistsError(f"Package directory already exists: {package}")
        package.mkdir()
        try:
            copy_package_files(workspace, package, policy)
        except Exception:
            shutil.rmtree(package)
            raise
        results.append(package)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cloud_exports"))
    parser.add_argument("--prefix", default="RobotArm_vla_cloud_20261115")
    parser.add_argument("--archive", action="store_true", help="Also create tar.gz and sha256 files")
    args = parser.parse_args(argv)
    for package in build_directories(args.workspace, args.output_dir, args.prefix):
        print(package)
    if args.archive:
        for package in (args.output_dir / f"{args.prefix}_{policy}" for policy in ("pi0", "smolvla")):
            archive = args.output_dir / f"{package.name}.tar.gz"
            checksum = archive_package(package, archive)
            print(f"{archive} ({archive.stat().st_size / 1024 / 1024:.2f} MiB)")
            print(checksum)


if __name__ == "__main__":
    main()
