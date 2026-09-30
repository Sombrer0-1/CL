"""G2 L_cal from completed development runs. Does not rewrite v3 evidence."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now

DEFAULT_RUNS = [
    "fullmem_v4_g2_cifar10_s0",
    "fullmem_v4_g2_cifar100_s0",
    "fullmem_v4_g2_core50_nc_s0",
    "fullmem_v4_g2_core50_ni_s0",
    "fullmem_v4_g2_core50_nic_s0",
    "fullmem_v4_g2_endless_ic_s0",
    "fullmem_v4_g2_endless_il_s0",
    "fullmem_v4_g2_endless_wc_s0",
]


def _learning_times(run_dir: Path) -> list[float]:
    path = run_dir / "experience_metrics.csv"
    if not path.is_file():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = []
    for row in rows:
        raw = row.get("learning_s")
        if raw in (None, ""):
            continue
        out.append(float(raw))
    return out


def lcal_report() -> dict[str, Any]:
    root = repo_root()
    members = []
    for name in DEFAULT_RUNS:
        times = _learning_times(root / "runs" / name)
        if not times:
            continue
        members.append(
            {
                "run_id": name,
                "n_experiences": len(times),
                "median_learning_s": statistics.median(times),
                "mean_learning_s": statistics.mean(times),
                "min_learning_s": min(times),
                "max_learning_s": max(times),
            }
        )
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "definition": "L is per-experience training wall time from experience_metrics.learning_s (excludes eval/controller/setup).",
        "previous_placeholder_latency_th_s": 30.0,
        "note": "CIFAR100 S0 median L is far below the 30s placeholder, so factor_l saturates 'fast enough' on this board. Do not import v3 L_cal.",
        "members": members,
        "adopted_for_o_recon_cifar100_s": next(
            (m["median_learning_s"] for m in members if m["run_id"] == "fullmem_v4_g2_cifar100_s0"),
            None,
        ),
        "adopted_for_o_recon_core50_nc_s": next(
            (m["median_learning_s"] for m in members if m["run_id"] == "fullmem_v4_g2_core50_nc_s0"),
            None,
        ),
        "adopted_for_o_recon_core50_nic_s": next(
            (m["median_learning_s"] for m in members if m["run_id"] == "fullmem_v4_g2_core50_nic_s0"),
            None,
        ),
    }
    dest = assert_write_path(root / "reports" / STUDY_ID / "g2" / "l_cal.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload
