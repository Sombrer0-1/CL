"""G2 physical-mapping and original-formula warning diagnostics."""

from __future__ import annotations

from typing import Any

from orion_repro.control.urge import urge_factors
from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now


def memory_sigmoid_table(m_max_mib: float, km: float = 0.25) -> list[dict[str, Any]]:
    rows = []
    slacks = [0, 8, 16, 32, 64, 256, 1024, 4096, int(m_max_mib)]
    for slack in slacks:
        m_t = max(0.0, float(m_max_mib) - float(slack))
        factors = urge_factors(
            plasticity=0.5,
            stability=0.5,
            latency_s=1.0,
            memory_mib=m_t,
            kp=0.25,
            ks=0.25,
            kl=0.25,
            km=km,
            p_th=0.5,
            s_th=0.5,
            latency_th_s=1.0,
            m_max_mib=m_max_mib,
        )
        rows.append(
            {
                "slack_mib": slack,
                "M_t_mib": m_t,
                "M_max_mib": m_max_mib,
                "factor_m": factors["factor_m"],
                "urge_with_other_neutral": factors["urge"],
            }
        )
    return rows


def run_physical_audit() -> dict[str, Any]:
    identity = collect_identity(include_cuda=True)
    mem_total_kb = int((identity.get("meminfo_kb") or {}).get("MemTotal") or 0)
    m_max_mib = mem_total_kb / 1024.0
    table = memory_sigmoid_table(m_max_mib)
    near_one = [row for row in table if row["factor_m"] >= 0.99]
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "identity": identity,
        "observation_protocol": {
            "M_max": "actual MemTotal, MiB",
            "M_t": "training-window peak MemTotal-MemAvailable, MiB",
            "g1_gpu_response": "confirmed in reports/fullmem_v4/admission/20260920T133630Z/shared_pool.json mixed gpu_drop_kb≈1.01GiB",
            "not_used_as_sum": ["host RSS", "cuda allocated", "cgroup"],
        },
        "formula_warning": {
            "km": 0.25,
            "note": "When M_t<=M_max, factor_m>=0.5. With km=0.25, 32MiB slack already drives factor_m near 1. Experience-boundary control cannot warn inside a step. This is O-recon behavior, not a silent O-eng guard.",
            "table": table,
            "slacks_with_factor_m_ge_0.99": [row["slack_mib"] for row in near_one],
        },
        "code_audit": {
            "m_batch_m_frame": "S0/static still ship count-space m_batch=m_frame=1.0. Complete O-recon uses paper 32x32 uint8 bytes (3072) for Eq.3-4; fitted peak slopes remain O-eng diagnosis only.",
            "controller_observation": "O-recon controlled_resource=board is training-window peak MemTotal-MemAvailable. device/host remain v3-compatible side channels, not O-recon M_t.",
            "replay": "ExperienceBalancedBuffer + storage_policy.resize; expand does not restore discarded samples (A09).",
            "optional_plugins": "TogglePlugin pause-keep-state; bytes still resident while disabled (toggles.py).",
            "gem_agem_gss": "patterns_per_experience = capacity // n_experiences; GEM/AGEM refuse stacked GEM (builder.py).",
            "prefetch": "bounded queue; H1/H3 later share prefetch on as system base, H7 isolates benefit.",
            "single_pass": "train each experience once (A03).",
        },
        "board": {
            "capacity_id": (identity.get("capacity") or {}).get("capacity_id"),
            "mem_total_mib": m_max_mib,
            "requested_mem": (identity.get("capacity") or {}).get("requested_mem"),
        },
    }
    dest = assert_write_path(repo_root() / "reports" / STUDY_ID / "g2" / "physical_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload
