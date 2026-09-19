"""Raw-evidence closure, offline URGE factor diagnostics, and DYN window audit.

These checks compare probe-manifest rows to run artifacts. They do not select
formal parameters and do not declare Orion effective.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from orion_repro.control.urge import urge_factors
from orion_repro.stages.effectiveness_v3.constants import PRIMARY_DATASETS, STUDY_ID
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.util import atomic_write_json, sha256_file


ARTIFACT_FILES = (
    "summary.json",
    "phase_trace.csv",
    "experience_metrics.csv",
    "resolved_config.yaml",
    "control_trace.jsonl",
    "events.jsonl",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def close_probe_row(row: dict[str, Any], context: StageContext) -> dict[str, Any]:
    """Cross-check one manifest row against the run directory, not just stored hashes."""
    probe_id = str(row.get("probe_id") or row.get("config_id") or "")
    role = str(row.get("role") or "")
    result: dict[str, Any] = {
        "probe_id": probe_id,
        "role": role,
        "dataset": row.get("dataset"),
        "run_id": row.get("run_id"),
        "status": row.get("status"),
        "ok": True,
        "mismatches": [],
    }
    if role == "host_capability":
        result["ok"] = row.get("status") in {"available", "unavailable"}
        if not result["ok"]:
            result["mismatches"].append("host_capability missing terminal status")
        return result
    if row.get("status") in {None, "not_run", "interrupted"}:
        result["ok"] = False
        result["mismatches"].append(f"non-terminal status {row.get('status')!r}")
        return result
    run_id = row.get("run_id")
    if not run_id:
        result["ok"] = False
        result["mismatches"].append("missing run_id")
        return result
    run_dir = context.root / "runs" / str(run_id)
    if not run_dir.is_dir():
        result["ok"] = False
        result["mismatches"].append(f"run directory missing: runs/{run_id}")
        return result
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        result["ok"] = False
        result["mismatches"].append("summary.json missing")
        return result
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != row.get("status"):
        result["mismatches"].append(
            f"status manifest={row.get('status')!r} summary={summary.get('status')!r}"
        )
    for name in ("n_trained", "n_evaluated"):
        summary_key = "n_experiences_trained" if name == "n_trained" else "n_experiences_evaluated"
        if row.get(name) not in (None, "") and summary.get(summary_key) not in (None, ""):
            if int(row[name]) != int(summary[summary_key]):
                result["mismatches"].append(f"{name} manifest={row.get(name)} summary={summary.get(summary_key)}")
    stored = row.get("artifact_hashes") or {}
    live: dict[str, str] = {}
    for name in ARTIFACT_FILES:
        path = run_dir / name
        if path.is_file():
            live[name] = sha256_file(path)
            expected = stored.get(name)
            if expected and expected != live[name]:
                result["mismatches"].append(f"hash mismatch {name}")
        elif name in stored:
            result["mismatches"].append(f"recorded artifact missing on disk: {name}")
    result["live_artifact_hashes"] = live
    phases = _read_csv(run_dir / "phase_trace.csv")
    train_exps = sorted(
        {
            int(p["experience_index"])
            for p in phases
            if p.get("phase") == "training" and p.get("experience_index") not in (None, "")
        }
    )
    if row.get("n_trained") not in (None, "") and train_exps:
        if int(row["n_trained"]) != len(train_exps) and summary.get("status") == "completed":
            result["mismatches"].append(
                f"phase_trace training experiences {len(train_exps)} != n_trained {row.get('n_trained')}"
            )
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.is_file():
        from orion_repro.stages.effectiveness_v3.util import load_yaml

        resolved = load_yaml(resolved_path)
        for key, field in (
            ("source_hash", "code_revision_or_snapshot"),
            ("environment_hash", "environment_lock_sha256"),
            ("split_hash", "dataset_manifest_sha256"),
        ):
            live_val = resolved.get(field)
            stored_val = row.get(key)
            if stored_val and live_val and stored_val != live_val:
                result["mismatches"].append(f"{key} mismatch vs resolved_config")
        fb = (resolved.get("controller") or {}).get("feedback_source")
        if fb and fb != "development_val_seen":
            result["mismatches"].append(f"resolved feedback_source={fb!r}")
        if row.get("feedback_source") != "development_val_seen":
            result["mismatches"].append(f"manifest feedback_source={row.get('feedback_source')!r}")
    result["ok"] = not result["mismatches"]
    return result


def close_raw_evidence(manifest: dict[str, Any], context: StageContext) -> dict[str, Any]:
    rows = []
    for row in manifest.get("runs") or []:
        rows.append(close_probe_row(row, context))
    n_ok = sum(1 for item in rows if item.get("ok"))
    payload = {
        "kind": "raw_evidence_closure",
        "study_id": STUDY_ID,
        "revision": context.revision,
        "n_rows": len(rows),
        "n_ok": n_ok,
        "n_mismatch": len(rows) - n_ok,
        "ok": all(item.get("ok") for item in rows) if rows else False,
        "rows": rows,
        "note": "ok means artifacts match the probe row; it is not an effectiveness verdict",
    }
    return payload


def write_evidence_closure(manifest: dict[str, Any], context: StageContext) -> Path:
    payload = close_raw_evidence(manifest, context)
    path = context.revision_dir / "evidence_closure.json"
    context.assert_output_path(path)
    atomic_write_json(path, payload)
    return path


def _l_cal_from_manifest(manifest: dict[str, Any], dataset: str) -> float | None:
    from statistics import median

    learning: list[float] = []
    for row in manifest.get("runs") or []:
        if row.get("role") != "l_cal" or row.get("dataset") != dataset:
            continue
        if row.get("status") != "completed":
            continue
        learning.extend(float(v) for v in (row.get("learning_s_by_experience") or []))
    return float(median(learning)) if learning else None


def factor_diagnostics(manifest: dict[str, Any], context: StageContext) -> dict[str, Any]:
    """Offline Eq.(1) factors from recorded P/S/L/M. Does not retune formal O11."""
    datasets: dict[str, Any] = {}
    for dataset in PRIMARY_DATASETS:
        l_cal = _l_cal_from_manifest(manifest, dataset) or 30.0
        items = []
        for row in manifest.get("runs") or []:
            if row.get("dataset") != dataset or not row.get("run_id"):
                continue
            if row.get("status") != "completed":
                continue
            run_dir = context.root / "runs" / str(row["run_id"])
            metrics = _read_csv(run_dir / "experience_metrics.csv")
            control = _read_jsonl(run_dir / "control_trace.jsonl")
            logged = {int(c["t"]): c.get("factors") for c in control if c.get("t") not in (None, "")}
            quota = row.get("quota_mib")
            m_th = float(quota) if quota not in (None, "") else 4096.0
            per_exp = []
            for metric in metrics:
                if metric.get("p_diag") in (None, "") or metric.get("s_initial") in (None, ""):
                    continue
                try:
                    k = int(metric["experience_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                p = float(metric["p_diag"])
                s = float(metric["s_initial"])
                lat = float(metric["learning_s"] or 0.0)
                mem = float(metric["memory_mib"] or 0.0)
                offline = urge_factors(
                    p,
                    s,
                    lat,
                    mem,
                    kp=0.25,
                    ks=0.25,
                    kl=0.25,
                    km=0.25,
                    p_th=0.5,
                    s_th=0.5,
                    latency_th_s=float(l_cal),
                    m_max_mib=m_th,
                )
                per_exp.append(
                    {
                        "experience": k,
                        "p_diag": p,
                        "s_initial": s,
                        "learning_s": lat,
                        "memory_mib": mem,
                        "offline_factors": offline,
                        "logged_factors": logged.get(k),
                    }
                )
            if per_exp:
                items.append(
                    {
                        "probe_id": row.get("probe_id"),
                        "role": row.get("role"),
                        "run_id": row.get("run_id"),
                        "n_experiences": len(per_exp),
                        "experiences": per_exp,
                    }
                )
        datasets[dataset] = {
            "latency_th_s": float(l_cal),
            "n_runs": len(items),
            "runs": items,
        }
    payload = {
        "kind": "factor_diagnostics",
        "study_id": STUDY_ID,
        "revision": context.revision,
        "note": "Offline four-factor product from recorded metrics; not a search over formal test P/S",
        "datasets": datasets,
    }
    path = context.revision_dir / "factor_diagnostics.json"
    context.assert_output_path(path)
    atomic_write_json(path, payload)
    return payload


def dyn_r_mib(row: dict[str, Any]) -> int:
    reserved = row.get("reserved_bytes_by_experience") or []
    highs = [int(x) for x in reserved if int(x or 0) > 0]
    if not highs:
        return 0
    return int(max(highs) // (1024**2))


def dyn_window_from_run(row: dict[str, Any], context: StageContext) -> dict[str, Any]:
    """Split S04 into mutation round, post-feedback window, and recovery."""
    reserved = [int(x or 0) for x in (row.get("reserved_bytes_by_experience") or [])]
    high = [i for i, value in enumerate(reserved) if value > 0]
    recovery = [i for i, value in enumerate(reserved) if value == 0 and high and i > min(high)]
    mutation = high[0] if high else None
    run_dir = context.root / "runs" / str(row["run_id"]) if row.get("run_id") else None
    events = _read_jsonl(run_dir / "events.jsonl") if run_dir else []
    control = _read_jsonl(run_dir / "control_trace.jsonl") if run_dir else []
    trans = [e for e in events if e.get("phase") == "resource_transition"]
    mutation_event = next((e for e in trans if mutation is not None and int(e.get("experience", -1)) == mutation), None)
    transition_ok = bool(mutation_event and mutation_event.get("status") == "ok")
    if row.get("failure_phase") == "resource_transition":
        transition_ok = False
    applied = []
    for item in control:
        t = item.get("t")
        if t in (None, ""):
            continue
        t = int(t)
        last_only = item.get("reason") == "last_experience" or item.get("applied_new_batch") is None
        changed = False
        if not last_only:
            if item.get("applied_new_batch") not in (None, row.get("new_batch")):
                changed = True
            if item.get("applied_replay") not in (None, row.get("replay_capacity")):
                changed = True
            if item.get("applied_optimizer_mode") == "advanced":
                changed = True
        applied.append({"t": t, "changed": changed, "last_experience_only": last_only})
    post_feedback = [a for a in applied if mutation is not None and a["t"] >= mutation and not a["last_experience_only"]]
    recovery_applied = [a for a in applied if a["t"] in set(recovery)]
    n_trained = int(row.get("n_trained") or 0)
    window = bool(transition_ok and post_feedback and n_trained > (mutation or 0) + 1)
    if row.get("status") != "completed" and not transition_ok:
        construction = "failed_to_construct"
        limitation = "mutation_round_failed"
    elif row.get("status") == "completed" and not window:
        construction = "failed_to_construct"
        limitation = "no_observable_post_feedback_window"
    elif window:
        construction = "realized"
        limitation = None
    else:
        construction = "not_run"
        limitation = "incomplete_dyn_evidence"
    return {
        "probe_id": row.get("probe_id"),
        "run_id": row.get("run_id"),
        "status": row.get("status"),
        "r_mib": dyn_r_mib(row),
        "high_experiences": high,
        "recovery_experiences": recovery,
        "mutation_experience": mutation,
        "transition_first_ok": transition_ok,
        "post_feedback_events": post_feedback,
        "recovery_events": recovery_applied,
        "observable_feedback_window": window,
        "construction_status": construction,
        "limitation": limitation,
        "n_trained": n_trained,
        "failure_phase": row.get("failure_phase"),
    }


def audit_dyn_windows(manifest: dict[str, Any], context: StageContext) -> dict[str, Any]:
    windows = []
    for row in manifest.get("runs") or []:
        if row.get("role") not in {"dyn_reservation_probe", "dyn_min_positive", "identity_s04_stamp"}:
            continue
        if not row.get("run_id"):
            windows.append({"probe_id": row.get("probe_id"), "construction_status": "not_run", "status": row.get("status")})
            continue
        windows.append(dyn_window_from_run(row, context))
    realized = next((w for w in windows if w.get("construction_status") == "realized"), None)
    mutation_fail = [w for w in windows if w.get("limitation") == "mutation_round_failed"]
    payload = {
        "kind": "dyn_window_audit",
        "study_id": STUDY_ID,
        "revision": context.revision,
        "windows": windows,
        "realized": realized,
        "mutation_failures_retained": mutation_fail,
        "s04_construction": realized["construction_status"] if realized else (
            "failed_to_construct" if windows else "not_run"
        ),
        "note": "Transition OOM is mutation-round evidence, not a post-feedback control failure",
    }
    path = context.revision_dir / "dyn_window.json"
    context.assert_output_path(path)
    atomic_write_json(path, payload)
    return payload


def half_life_delta(n_experiences: int) -> float:
    if n_experiences <= 1:
        raise ValueError("half-life delta needs N>1")
    return float(math.log(2.0) / float(n_experiences - 1))
