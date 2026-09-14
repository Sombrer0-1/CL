"""Host memory enforcement probe. Does not change WSL/global config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def cgroup_memory_max() -> dict[str, Any]:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return {
                "available": False,
                "path": str(path),
                "reason": f"unreadable: {exc}",
                "mechanism": "cgroup_memory_max",
            }
        writable = os.access(path, os.W_OK)
        return {
            "available": raw not in {"max", ""} and raw.isdigit(),
            "writable": writable,
            "path": str(path),
            "value": raw,
            "mechanism": "cgroup_memory_max",
            "reason": None
            if writable
            else "process cannot write cgroup memory.max; host_enforced blocked",
        }
    return {
        "available": False,
        "writable": False,
        "path": None,
        "mechanism": "cgroup_memory_max",
        "reason": "no cgroup memory.max in this WSL namespace",
    }


def rlimit_as_supported() -> dict[str, Any]:
    try:
        import resource
    except ImportError:
        return {
            "available": False,
            "mechanism": "rlimit_as",
            "reason": "resource module unavailable",
        }
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "available": True,
        "mechanism": "rlimit_as",
        "soft": None if soft == resource.RLIM_INFINITY else int(soft),
        "hard": None if hard == resource.RLIM_INFINITY else int(hard),
        "scope": "process virtual address space, not RSS; CUDA mappings inflate VA",
        "reason": "usable as a named host_rlimit_as variant, not a cgroup RSS cap",
    }


def probe_host_enforcement() -> dict[str, Any]:
    cgroup = cgroup_memory_max()
    rlimit = rlimit_as_supported()
    host_enforced = bool(cgroup.get("writable"))
    return {
        "host_enforced": "available" if host_enforced else "blocked",
        "cgroup": cgroup,
        "rlimit_as": rlimit,
        "notes": (
            "PLAN host_enforced requires a verified process-group/cgroup limit. "
            "This probe never writes WSL global memory settings."
        ),
    }
