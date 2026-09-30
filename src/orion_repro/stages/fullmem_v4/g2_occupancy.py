"""Summarize G2 resource traces. Does not rewrite v3 evidence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from orion_repro.control.urge import urge_factors
from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, load_json, utc_now

RUNS = [
    "fullmem_v4_g2_cifar10_s0",
    "fullmem_v4_g2_cifar100_s0",
    "fullmem_v4_g2_cifar100_batch256",
    "fullmem_v4_g2_cifar100_replay2000",
    "fullmem_v4_g2_cifar100_er_gem",
    "fullmem_v4_g2_cifar100_er_ewc",
    "fullmem_v4_g2_cifar100_er_gem_ewc",
    "fullmem_v4_g2_core50_nc_s0",
    "fullmem_v4_g2_cifar100_prefetch",
    "fullmem_v4_g2_core50_ni_s0",
    "fullmem_v4_g2_core50_nic_s0",
    "fullmem_v4_g2_endless_ic_s0",
    "fullmem_v4_g2_endless_il_s0",
    "fullmem_v4_g2_endless_wc_s0",
]


def _nums(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def summarize_run(run_dir: Path, mem_total_bytes: int) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    trace_path = run_dir / "resource_trace.csv"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    row: dict[str, Any] = {
        "run_id": summary.get("run_id") or run_dir.name,
        "status": summary.get("status"),
        "method_id": summary.get("method_id"),
        "dataset": summary.get("dataset"),
        "n_experiences_run": summary.get("n_experiences_run"),
        "p_diag": summary.get("p_diag"),
        "s_initial": summary.get("s_initial"),
        "online_total_s": summary.get("online_total_s"),
        "run_total_s": summary.get("run_total_s"),
        "run_dir": str(run_dir),
    }
    if trace_path.is_file():
        with trace_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        avail = _nums(rows, "system_available_bytes")
        gpu = _nums(rows, "gpu_reserved_peak_bytes")
        rss = _nums(rows, "proc_rss_bytes")
        if avail:
            min_avail = min(avail)
            peak_used = mem_total_bytes - min_avail
            row["min_system_available_bytes"] = min_avail
            row["peak_memtotal_minus_available_bytes"] = peak_used
            row["peak_M_t_mib"] = peak_used / (1024.0 * 1024.0)
        if gpu:
            row["gpu_reserved_peak_bytes"] = max(gpu)
        if rss:
            row["proc_rss_peak_bytes"] = max(rss)
    return row


def occupancy_report() -> dict[str, Any]:
    identity = collect_identity(include_cuda=False)
    mem_total_kb = int((identity.get("meminfo_kb") or {}).get("MemTotal") or 0)
    mem_total_bytes = mem_total_kb * 1024
    m_max_mib = mem_total_kb / 1024.0
    root = repo_root()
    names = list(RUNS)
    runs_root = root / "runs"
    if runs_root.is_dir():
        names.extend(p.name for p in sorted(runs_root.glob("fullmem_v4_g2_*")) if p.is_dir())
    members = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        found = summarize_run(runs_root / name, mem_total_bytes)
        if found:
            members.append(found)
    slacks = []
    for item in members:
        mt = item.get("peak_M_t_mib")
        if mt is None:
            continue
        factors = urge_factors(
            plasticity=0.5,
            stability=0.5,
            latency_s=1.0,
            memory_mib=float(mt),
            kp=0.25,
            ks=0.25,
            kl=0.25,
            km=0.25,
            p_th=0.5,
            s_th=0.5,
            latency_th_s=1.0,
            m_max_mib=m_max_mib,
        )
        slack = m_max_mib - float(mt)
        slacks.append(
            {
                "run_id": item["run_id"],
                "slack_mib": slack,
                "factor_m": factors["factor_m"],
            }
        )
    cap_id = (identity.get("capacity") or {}).get("capacity_id") or "unknown"
    scoped = [item for item in members if cap_id != "unknown" and cap_id in str(item.get("run_id") or "")]
    pressure_members = scoped or members
    min_avail = [
        item.get("min_system_available_bytes")
        for item in pressure_members
        if item.get("min_system_available_bytes") is not None
    ]
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "board": {
            "capacity_id": cap_id,
            "mem_total_bytes": mem_total_bytes,
            "mem_total_mib": m_max_mib,
        },
        "pressure_scene": f"unestablished_at_{cap_id}",
        "pressure_scope": "run_id_contains_capacity" if scoped else "all_g2_runs_mixed",
        "note": (
            "Do not enlarge the model to manufacture pressure. "
            "A static pressure scene is established only if S0 train/eval completes twice "
            "and a pre-registered high-resource static shows measurable pressure or resource failure."
        ),
        "members": members,
        "pressure_run_ids": [item.get("run_id") for item in pressure_members],
        "factor_m_at_observed_peak": slacks,
        "min_available_bytes_across_completed": min(min_avail) if min_avail else None,
    }
    dest_dir = root / "reports" / STUDY_ID / "g2"
    dest = assert_write_path(dest_dir / "occupancy.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    cap_copy = assert_write_path(dest_dir / f"occupancy_{cap_id}.json")
    atomic_write_json(cap_copy, payload)
    h4_members = [item for item in members if "_h4_" in str(item.get("run_id") or "")]
    if h4_members:
        h4_avail = [
            item.get("min_system_available_bytes")
            for item in h4_members
            if item.get("min_system_available_bytes") is not None
        ]
        h4_payload = dict(payload)
        h4_payload["pressure_scope"] = "run_id_contains_h4"
        h4_payload["members"] = h4_members
        h4_payload["pressure_run_ids"] = [item.get("run_id") for item in h4_members]
        h4_payload["factor_m_at_observed_peak"] = [
            row for row in slacks if "_h4_" in str(row.get("run_id") or "")
        ]
        h4_payload["min_available_bytes_across_completed"] = min(h4_avail) if h4_avail else None
        h4_payload["note"] = (
            "H4 co-running occupancy only. Dedicated background process, not static tight. "
            "Do not treat as a G3 freeze. Do not enlarge the model."
        )
        h4_copy = assert_write_path(dest_dir / "occupancy_h4.json")
        atomic_write_json(h4_copy, h4_payload)
        payload["h4_copy"] = str(h4_copy)
        payload["n_h4_members"] = len(h4_members)
        nic_h4 = [item for item in h4_members if "_nic_" in str(item.get("run_id") or "")]
        if nic_h4:
            nic_avail = [
                item.get("min_system_available_bytes")
                for item in nic_h4
                if item.get("min_system_available_bytes") is not None
            ]
            nic_payload = dict(h4_payload)
            nic_payload["pressure_scope"] = "run_id_contains_h4_nic"
            nic_payload["members"] = nic_h4
            nic_payload["pressure_run_ids"] = [item.get("run_id") for item in nic_h4]
            nic_payload["factor_m_at_observed_peak"] = [
                row for row in (h4_payload.get("factor_m_at_observed_peak") or [])
                if "_nic_" in str(row.get("run_id") or "")
            ]
            nic_payload["min_available_bytes_across_completed"] = min(nic_avail) if nic_avail else None
            nic_copy = assert_write_path(dest_dir / "occupancy_h4_nic.json")
            atomic_write_json(nic_copy, nic_payload)
            payload["h4_nic_copy"] = str(nic_copy)
            payload["n_h4_nic_members"] = len(nic_h4)
    payload["output"] = str(dest)
    payload["capacity_copy"] = str(cap_copy)
    return payload
