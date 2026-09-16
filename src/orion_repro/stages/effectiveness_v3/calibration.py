"""Calibration rules for effectiveness_v3. Does not read formal P/S."""

from __future__ import annotations

import copy
import math
from typing import Any

from orion_repro.stages.effectiveness_v3.constants import (
    EVAL_BATCH_CANDIDATES,
    PRIMARY_DATASETS,
    QUOTA_GRID_MIB,
    STUDY_ID,
)
from orion_repro.stages.effectiveness_v3.development import N_EXPECTED
from orion_repro.stages.effectiveness_v3.schema import validate_probe_row
from orion_repro.stages.effectiveness_v3.util import StageError, canonical_hash


class CalibrationError(StageError):
    pass


def _median(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise CalibrationError("empty median")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _rows(manifest: dict[str, Any], *, role: str, dataset: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in manifest.get("runs") or []:
        if row.get("role") != role:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        out.append(row)
    return out


def _select_static(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [c for c in candidates if c.get("status") == "completed"]
    failed = [c for c in candidates if c.get("status") in {"cuda_oom", "host_oom", "budget_exceeded"}]
    errors = [c for c in candidates if c.get("status") == "implementation_error"]
    missing = [c for c in candidates if c.get("status") in {None, "not_run", "interrupted"}]
    if errors:
        raise CalibrationError("implementation errors must be fixed, not treated as resource infeasibility")
    if missing:
        raise CalibrationError(f"static candidates missing outcomes: {[c.get('config_id') for c in missing]}")
    if not feasible:
        raise CalibrationError("no feasible static candidates")
    ranked = sorted(
        feasible,
        key=lambda row: (
            -((float(row["p_diag"]) + float(row["s_initial"])) / 2.0),
            float(row["online_total_s"]),
            str(row.get("config_id") or row.get("config") or ""),
        ),
    )
    winner = ranked[0]
    return {
        "new_batch": int(winner["new_batch"]),
        "replay_capacity": int(winner["replay_capacity"]),
        "config_id": str(winner.get("config_id") or winner.get("config")),
        "run_id": winner.get("run_id"),
        "score": (float(winner["p_diag"]) + float(winner["s_initial"])) / 2.0,
        "n_feasible": len(feasible),
        "n_resource_failed": len(failed),
        "rule": "max (P+S)/2; then shorter online_total_s; then lexical config_id",
    }


def _eval_batch(rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise CalibrationError("missing eval_batch evidence")
    for batch in EVAL_BATCH_CANDIDATES:
        hits = [r for r in rows if int(r.get("eval_batch") or 0) == batch]
        if hits and all(h.get("status") == "completed" for h in hits):
            return int(batch)
    raise CalibrationError("no feasible eval_batch among 128/32/8")


def _quota_choice(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    if not rows:
        raise CalibrationError("missing quota_scan evidence")
    evidence = []
    chosen = None
    for row in sorted(rows, key=lambda r: int(r["quota_mib"])):
        status = row.get("status")
        ratio = row.get("train_reserved_peak_ratio")
        item = {
            "quota_mib": int(row["quota_mib"]),
            "status": status,
            "train_reserved_peak_ratio": ratio,
            "run_id": row.get("run_id"),
        }
        evidence.append(item)
        if status != "completed" or ratio is None:
            continue
        if 0.80 <= float(ratio) <= 0.95 and chosen is None:
            chosen = row
    if chosen is None:
        raise CalibrationError("no Q_tight with completed S0 train/eval and reserved peak in 80-95% of quota")
    return int(chosen["quota_mib"]), {"scans": evidence, "selected_tight_run": chosen.get("run_id")}


def mid_quota_mib(tight: int, loose: int, grid: int = QUOTA_GRID_MIB) -> int:
    if loose <= tight:
        raise CalibrationError("Q_loose must exceed Q_tight")
    candidates = list(range(tight + grid, loose, grid))
    if not candidates:
        raise CalibrationError("no interior mid-grid point between Q_tight and Q_loose")
    target = (tight + loose) / 2.0
    return min(candidates, key=lambda value: (abs(value - target), value))


def _require_full_stream(row: dict[str, Any], dataset: str) -> None:
    expected = int(row.get("n_expected") or N_EXPECTED[dataset])
    if not row.get("full_stream", True):
        raise CalibrationError(f"{row.get('config_id')} is not a full-stream probe")
    if int(row.get("n_trained") or -1) != expected or int(row.get("n_evaluated") or -1) != expected:
        raise CalibrationError(
            f"{row.get('config_id')} incomplete stream trained={row.get('n_trained')} "
            f"evaluated={row.get('n_evaluated')} expected={expected}"
        )


def calibrate(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("study_id") != STUDY_ID:
        raise CalibrationError("probe_manifest study_id must be effectiveness_v3")
    rows = list(manifest.get("runs") or [])
    if not rows:
        raise CalibrationError("probe_manifest has no runs")
    for row in rows:
        if row.get("role") == "host_capability":
            continue
        try:
            validate_probe_row(row)
        except StageError as exc:
            raise CalibrationError(str(exc)) from exc
    datasets: dict[str, Any] = {}
    for dataset in PRIMARY_DATASETS:
        eval_rows = _rows(manifest, role="eval_batch", dataset=dataset)
        eval_batch = _eval_batch(eval_rows)
        quota_rows = _rows(manifest, role="quota_scan", dataset=dataset)
        q_tight, quota_evidence = _quota_choice(quota_rows)
        loose_rows = _rows(manifest, role="q_loose_verify", dataset=dataset)
        q_loose = 2 * q_tight
        if loose_rows:
            ok = [r for r in loose_rows if r.get("status") == "completed"]
            if not ok:
                raise CalibrationError(f"{dataset} Q_loose was not feasible")
            verified = [r for r in ok if int(r["quota_mib"]) == q_loose]
            if verified:
                q_loose = int(verified[0]["quota_mib"])
            else:
                q_loose = int(ok[0]["quota_mib"])
        l_rows = [r for r in _rows(manifest, role="l_cal", dataset=dataset) if r.get("status") == "completed"]
        if len(l_rows) < 2:
            raise CalibrationError(f"{dataset} L_cal requires two completed Q_loose S0 runs")
        learning: list[float] = []
        for row in l_rows:
            _require_full_stream(row, dataset)
            values = row.get("learning_s_by_experience")
            if not values:
                raise CalibrationError(f"{dataset} L_cal run missing per-experience learning_s")
            learning.extend(float(v) for v in values)
        l_cal = _median(learning)
        s_tight_rows = _rows(manifest, role="static_search_tight", dataset=dataset)
        s_loose_rows = _rows(manifest, role="static_search_loose", dataset=dataset)
        static_tight = _select_static(s_tight_rows) if s_tight_rows else None
        static_loose = _select_static(s_loose_rows) if s_loose_rows else None
        if static_tight is None or static_loose is None:
            raise CalibrationError(f"{dataset} static search incomplete")
        try:
            q_mid = mid_quota_mib(q_tight, q_loose)
            mid_status = "available"
        except CalibrationError as exc:
            q_mid = None
            mid_status = str(exc)
        datasets[dataset] = {
            "eval_batch": int(eval_batch),
            "quota_tight_mib": int(q_tight),
            "quota_loose_mib": int(q_loose),
            "quota_mid_mib": q_mid,
            "quota_mid_status": mid_status,
            "quota_tight_bytes": int(q_tight) * 1024**2,
            "quota_loose_bytes": int(q_loose) * 1024**2,
            "latency_cal_s": float(l_cal),
            "static_selections": {"tight": static_tight, "loose": static_loose},
            "quota_evidence": quota_evidence,
            "l_cal_run_ids": [r.get("run_id") for r in l_rows],
        }
    dyn_search = _rows(manifest, role="static_search_dyn", dataset="core50_nc")
    s_dyn = _select_static(dyn_search) if dyn_search else copy.deepcopy(datasets["core50_nc"]["static_selections"]["tight"])
    datasets["core50_nc"]["static_selections"]["dyn"] = s_dyn
    control = _rows(manifest, role="control_2x2", dataset="core50_nc")
    identifiable = any(r.get("status") == "completed" and r.get("integer_config_changed") for r in control)
    dyn_probe = _rows(manifest, role="dyn_reservation_probe", dataset="core50_nc")
    reserved = [0] * 9
    dyn_constructed = False
    dyn_quota_bytes = datasets["core50_nc"]["quota_tight_bytes"]
    if dyn_probe:
        ok = next((r for r in dyn_probe if r.get("status") == "completed" and r.get("transition_first_ok")), None)
        if ok:
            reserved = [int(x) for x in ok["reserved_bytes_by_experience"]]
            dyn_constructed = True
            if ok.get("quota_mib"):
                dyn_quota_bytes = int(ok["quota_mib"]) * 1024**2
    io_off = next((r for r in _rows(manifest, role="io_off", dataset="core50_nc") if r.get("status") == "completed"), None)
    io_ratio = None
    io_constructed = False
    if io_off is not None:
        io_ratio = io_off.get("supply_wait_ratio")
        io_constructed = io_ratio is not None and float(io_ratio) >= 0.10
    host = next((r for r in rows if r.get("role") == "host_capability"), None)
    host_status = "unavailable"
    if host:
        host_status = str(host.get("status") or "unavailable")
    coverage = {
        "S01": "pending_formal" if identifiable else "failed_to_construct",
        "S02": "pending_formal",
        "S03": "pending_formal",
        "S04": "realized" if dyn_constructed else "failed_to_construct",
        "S05": "pending_formal",
        "S06": "realized" if identifiable else "failed_to_construct",
        "S07": "realized" if io_constructed else "failed_to_construct",
        "S08": "pending_formal",
        "eval_separation": "realized",
        "control_identifiable": "realized" if identifiable else "failed_to_construct",
        "dynamic_reservation": "realized" if dyn_constructed else "failed_to_construct",
        "io_prefetch": "realized" if io_constructed else "failed_to_construct",
        "host": host_status,
    }
    result = {
        "kind": "calibration_result",
        "study_id": STUDY_ID,
        "datasets": datasets,
        "thr0": 0.05,
        "delta": 0.0,
        "alpha": 0.1,
        "beta": 0.2,
        "initial_new_batch": 16,
        "initial_replay": 200,
        "optional_plugins": "gem_ewc",
        "m_batch": 1.0,
        "m_frame": 1.0,
        "cost_model_note": "count-space proxy m_batch=m_frame=1.0; not physical MiB",
        "reserved_bytes_by_experience": reserved,
        "dyn_quota_bytes": int(dyn_quota_bytes),
        "control_identifiable": bool(identifiable),
        "scenario_coverage": coverage,
        "io_prefetch": {
            "constructed": bool(io_constructed),
            "wait_ratio": float(io_ratio) if io_ratio is not None else 0.0,
            "pin_memory": False,
            "queue_depth": 2,
            "num_workers": 0,
        },
        "host": {"status": host_status, "row": host},
        "source_runs": [r.get("run_id") for r in rows if r.get("run_id")],
        "probe_manifest_hash": canonical_hash(manifest),
        "o00_latency_s": 30.0,
        "o00_memory_mib": 4096.0,
        "decay_delta": math.log(2.0) / 8.0,
    }
    result["calibration_hash"] = canonical_hash({k: v for k, v in result.items() if k != "calibration_hash"})
    return result
