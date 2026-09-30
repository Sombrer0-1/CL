"""G2 cost sketch from measured wall times. Not an executable formal queue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.g2_occupancy import summarize_run
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now

# PLAN §6 design slots (before dedup). 3 capacities are a design ceiling, not admitted.
H_BLOCKS = {
    "H1": 315,
    "H2": 108,
    "H3": 189,
    "H4": 48,
    "H5": 180,
    "H6": 180,
    "H7": 120,
    "H8": 180,
}


def cost_estimate() -> dict[str, Any]:
    identity = collect_identity(include_cuda=False)
    root = repo_root()
    members = []
    for path in sorted((root / "runs").glob("fullmem_v4_g2_*")):
        if not path.is_dir():
            continue
        row = summarize_run(path, int((identity.get("meminfo_kb") or {}).get("MemTotal") or 0) * 1024)
        if row:
            members.append(row)
    by_id = {m["run_id"]: m for m in members}

    def _wall(*names: str, default: float) -> float:
        for name in names:
            raw = (by_id.get(name) or {}).get("run_total_s")
            if raw not in (None, ""):
                return float(raw)
        return default

    s0 = _wall(
        "fullmem_v4_g2_cifar100_s0_seed17_mem16g",
        "fullmem_v4_g2_cifar100_s0_hash_off_seed17",
        "fullmem_v4_g2_cifar100_s0",
        default=68.0,
    )
    heavy = _wall(
        "fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem16g",
        "fullmem_v4_g2_cifar100_er_gem_ewc_seed17",
        "fullmem_v4_g2_cifar100_er_gem_ewc",
        default=330.0,
    )
    nic = _wall(
        "fullmem_v4_g2_core50_nic_s0_seed17_mem16g",
        "fullmem_v4_g2_core50_nic_s0_seed17",
        "fullmem_v4_g2_core50_nic_s0",
        default=1750.0,
    )
    nic_stack = _wall(
        "fullmem_v4_g2_core50_nic_er_gem_ewc_seed17_mem16g",
        default=6882.0,
    )
    design_n = sum(H_BLOCKS.values())
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "not_a_freeze": True,
        "admitted_capacity": ["mem64g", "mem32g", "mem16g"],
        "unestablished_capacity": [],
        "pressure_scene": "unestablished_at_mem16g",
        "pressure_note": "All three admitted mem= tiers left GiB-scale slack on ResNet-20 32x32. Admission is not a freeze. factor_m stayed 1.0.",
        "design_slots_before_dedup": design_n,
        "blocks": H_BLOCKS,
        "measured_run_total_s": {m["run_id"]: m.get("run_total_s") for m in members},
        "sketch": {
            "cifar100_s0_s": s0,
            "cifar100_gem_ewc_s": heavy,
            "core50_nic_s0_s": nic,
            "core50_nic_er_gem_ewc_mem16g_s": nic_stack,
            "if_all_1320_at_cifar100_s0_hours": design_n * s0 / 3600.0,
            "if_all_1320_at_gem_ewc_hours": design_n * heavy / 3600.0,
            "h1_five_streams_one_capacity_three_seeds_at_s0_hours": 5 * 7 * 1 * 3 * s0 / 3600.0,
            "oracle_42_cifar100_at_s0_hours": 42 * s0 / 3600.0,
            "oracle_1008_design_at_s0_hours": 1008 * s0 / 3600.0,
            "note": "Lower bound ignores Oracle search, GSS/GEM/AGEM, eval, failures, and H3 Endless official protocol. NIC S0 is ~29 min; NIC ER+GEM+EWC on mem16g was 6882 s. 16GiB is admitted, not frozen; pressure still unestablished.",
        },
        "executable_now": "kind=probe development only; kind=train refused until G3; mem64g/mem32g/mem16g admitted, none frozen; pressure unestablished",
    }
    dest = assert_write_path(root / "reports" / STUDY_ID / "g2" / "cost_estimate.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload
