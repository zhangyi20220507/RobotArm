"""Download and install the SAPIEN PhysX GPU backend library.

Use this when ManiSkill/SAPIEN prints a message like:
  Downloading PhysX GPU library to ~/.sapien/physx/... from Github

Install / verify:
  robotarm simulation setup-physx --

Force reinstall after an interrupted download:
  robotarm simulation setup-physx -- --force

Expected installed file:
  /root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so

After this succeeds, GPU simulation can be used with:
  --sim-backend gpu
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_VERSION = "105.1-physx-5.3.1.patch0"
DEFAULT_URL = (
    "https://github.com/sapien-sim/physx-precompiled/releases/download/"
    f"{DEFAULT_VERSION}/linux-so.zip"
)
REQUIRED_LIB = "libPhysXGpu_64.so"


def download(url: str, dst: Path, chunk_size: int = 1024 * 1024):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as response, dst.open("wb") as f:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        done = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                print(f"\rDownloading {pct:5.1f}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="")
            else:
                print(f"\rDownloading {done / 1e6:.1f} MB", end="")
    print()


def install(url: str, target: Path, force: bool = False):
    target.mkdir(parents=True, exist_ok=True)
    lib_path = target / REQUIRED_LIB
    if lib_path.exists() and not force:
        print(f"Already installed: {lib_path}")
        return lib_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "linux-so.zip"
        extract_dir = tmpdir / "extract"
        extract_dir.mkdir()

        print(f"Downloading from:\n  {url}")
        print(f"Target directory:\n  {target}")
        download(url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        found = list(extract_dir.rglob(REQUIRED_LIB))
        if not found:
            raise FileNotFoundError(
                f"{REQUIRED_LIB} was not found inside {zip_path}. "
                "The archive may be incomplete or not the expected Linux PhysX package."
            )

        if force:
            for old_file in target.glob("*"):
                if old_file.is_file() or old_file.is_symlink():
                    old_file.unlink()
                elif old_file.is_dir():
                    shutil.rmtree(old_file)

        for item in extract_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(extract_dir)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

    if not lib_path.exists():
        copied = list(target.rglob(REQUIRED_LIB))
        if copied:
            shutil.copy2(copied[0], lib_path)
        else:
            raise FileNotFoundError(f"Install finished but {lib_path} is missing")

    print(f"Installed: {lib_path}")
    return lib_path


def verify():
    print("Verifying SAPIEN PhysX GPU...")
    import sapien.physx as physx

    physx.enable_gpu()
    print("SAPIEN PhysX GPU enabled:", physx.is_gpu_enabled())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--url", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--force", action="store_true", help="Reinstall even if the library already exists.")
    parser.add_argument("--no-verify", action="store_true", help="Only download/extract, do not call physx.enable_gpu().")
    args = parser.parse_args()

    url = args.url or (
        "https://github.com/sapien-sim/physx-precompiled/releases/download/"
        f"{args.version}/linux-so.zip"
    )
    target = Path(args.target).expanduser() if args.target else Path.home() / ".sapien" / "physx" / args.version

    try:
        install(url, target, force=args.force)
        if not args.no_verify:
            verify()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("\nManual fallback:")
        print(f"  1. Download: {url}")
        print(f"  2. Unzip it into: {target}")
        print(f"  3. Confirm this file exists: {target / REQUIRED_LIB}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
