"""Verify the packaged source/data/weights before cloud installation."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "CLOUD_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        path = root / relative
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {relative}")
    print(f"PASS: {len(manifest['files'])} files; {manifest['dataset']['episodes']} episodes; "
          f"{manifest['dataset']['frames']} frames; source/data/base/tokenizer verified")


if __name__ == "__main__":
    main()
