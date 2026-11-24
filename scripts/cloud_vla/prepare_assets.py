"""Download the policy base and tokenizer/VLM assets into a bundle-local cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICIES = {
    "pi0": {
        "base_repo": "lerobot/pi0_base",
        "base_dir": "artifacts/pretrained/pi0_base",
        "dependencies": ("google/paligemma-3b-pt-224",),
    },
    "smolvla": {
        "base_repo": "lerobot/smolvla_base",
        "base_dir": "artifacts/pretrained/smolvla_base",
        "dependencies": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",),
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    root = args.bundle_root.expanduser().resolve()
    spec = POLICIES[args.policy]
    base_dir = root / spec["base_dir"]
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec['base_repo']} -> {base_dir}", flush=True)
    snapshot_download(
        repo_id=spec["base_repo"],
        revision=args.revision,
        local_dir=base_dir,
    )
    for repo_id in spec["dependencies"]:
        print(f"Caching dependency: {repo_id}", flush=True)
        snapshot_download(repo_id=repo_id)

    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != args.policy or not (base_dir / "model.safetensors").is_file():
        raise SystemExit(f"Downloaded base is incomplete or has the wrong type: {base_dir}")
    print(f"Assets ready for {args.policy}: {base_dir}")


if __name__ == "__main__":
    main()
