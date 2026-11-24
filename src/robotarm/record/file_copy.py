"""Copy-on-write file copy for immutable training data with safe fallback."""

from __future__ import annotations

import errno
import fcntl
import shutil


FICLONE = 0x40049409


def clone_or_copy(source: str, target: str) -> str:
    """Use a filesystem reflink if available; otherwise copy bytes."""
    try:
        with open(source, "rb") as src, open(target, "wb") as dst:
            fcntl.ioctl(dst.fileno(), FICLONE, src.fileno())
    except OSError as exc:
        if exc.errno not in (errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL):
            raise
        shutil.copyfile(source, target)
    shutil.copystat(source, target)
    return target
