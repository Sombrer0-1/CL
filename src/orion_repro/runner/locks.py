"""Exclusive project lock so light24 and pressure_v2 cannot train together."""

from __future__ import annotations

import fcntl
from pathlib import Path


def acquire_locks(paths: list[Path]):
    handles = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(str(path))
            handle.flush()
            handles.append(handle)
    except BlockingIOError:
        for handle in handles:
            handle.close()
        raise
    return handles
