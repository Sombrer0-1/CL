"""Idle MemAvailable windows for G3 background admission. Not a freeze."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now


def _meminfo_kb() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        token = rest.strip().split()[0]
        try:
            out[key] = int(token)
        except ValueError:
            continue
    return out


def idle_baseline(*, windows: int = 3, seconds: int = 60, interval_s: float = 1.0) -> dict[str, Any]:
    identity = collect_identity(include_cuda=False)
    rows = []
    for i in range(windows):
        samples = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            info = _meminfo_kb()
            samples.append(
                {
                    "ts": utc_now(),
                    "MemAvailable_kb": info.get("MemAvailable"),
                    "MemFree_kb": info.get("MemFree"),
                    "Cached_kb": info.get("Cached"),
                }
            )
            time.sleep(interval_s)
        avail = [s["MemAvailable_kb"] for s in samples if s.get("MemAvailable_kb") is not None]
        rows.append(
            {
                "window": i,
                "n_samples": len(samples),
                "min_MemAvailable_kb": min(avail) if avail else None,
                "max_MemAvailable_kb": max(avail) if avail else None,
                "mean_MemAvailable_kb": (sum(avail) / len(avail)) if avail else None,
            }
        )
        if i + 1 < windows:
            time.sleep(5)
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "boot_id": identity.get("boot_id"),
        "capacity": identity.get("capacity"),
        "windows": rows,
        "note": "Development idle windows on mem64g. Not a G3 freeze of admission ranges.",
    }
    dest = assert_write_path(repo_root() / "reports" / STUDY_ID / "g2" / "idle_baseline.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload
