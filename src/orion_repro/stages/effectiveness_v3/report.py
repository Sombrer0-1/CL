"""Matrix-driven reporting. Never glob-and-overwrite by method/seed.

Tables are left-joined from the matrix (or development probe list). A missing
or failed attempt stays in the row; survivors are not rewritten as the cell.
Effectiveness verdicts are not inferred from URGE triggers or completion counts.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from orion_repro.stages.effectiveness_v3.constants import STUDY_ID
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.coverage import audit_scenarios
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json, atomic_write_text, load_yaml


PAIR_CORE = (
    "dataset",
    "split_manifest",
    "split_hash",
    "model_name",
    "optimizer",
    "seed",
    "eval_batch",
    "precision",
    "quota_bytes",
    "source_hash",
    "environment_hash",
    "frozen_hash",
    "resource_sequence",
    "io_backend",
    "num_workers",
    "pin_memory",
    "transform_fingerprint",
    "platform_gpu",
)

PAIR_ALLOW_BY_CONTRAST = {
    ("O11", "S0"): {"method", "controller", "optional_plugins", "plugin_policy"},
    ("O11", "S_star"): {"method", "controller", "optional_plugins", "plugin_policy", "new_batch", "replay_capacity"},
    ("O11", "R11"): {"method", "plugin_policy", "optional_start_enabled"},
    ("O11", "F11"): {"method", "plugin_policy", "optional_start_enabled"},
    ("O11", "O00"): {"method", "latency_s", "m_max_mib"},
    ("O11", "O10"): {"method", "m_max_mib"},
    ("O11", "O01"): {"method", "latency_s"},
}

TABLE_NAMES = (
    "coverage.csv",
    "attempts.csv",
    "means.csv",
    "paired.csv",
    "phase_resources.csv",
    "control_events.csv",
    "plugin_activity.csv",
    "supply.csv",
    "search_cost.csv",
)


def collect(matrix: dict[str, Any], context: StageContext) -> dict[str, Any]:
    context.reject_foreign_study(matrix)
    progress = {"results": []}
    if context.progress_path.exists():
        progress = json.loads(context.progress_path.read_text(encoding="utf-8"))
    attempts = list(progress.get("results") or [])
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        cell = row.get("cell_id")
        if cell:
            by_cell[cell].append(row)
    joined = []
    for cell in matrix.get("cells") or []:
        cell_id = cell.get("cell_id")
        rows = by_cell.get(cell_id, [])
        joined.append({"cell": cell, "attempts": rows, "n_attempts": len(rows)})
    return {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "frozen_hash": matrix.get("frozen_hash"),
        "design_slots": matrix.get("design_slots"),
        "eligible_n": matrix.get("eligible_n"),
        "excluded_n": matrix.get("excluded_n"),
        "cells": joined,
        "attempts": attempts,
        "matrix": matrix,
    }


def validate_pairs(left: dict[str, Any], right: dict[str, Any], *, allow: set[str] | None = None) -> None:
    allow = allow or set()
    for key in PAIR_CORE:
        if key in allow:
            continue
        if left.get(key) != right.get(key):
            raise StageError(f"pairing mismatch on {key}: {left.get(key)!r} vs {right.get(key)!r}")
    if left.get("frozen_hash") and right.get("frozen_hash") and left["frozen_hash"] != right["frozen_hash"]:
        raise StageError("refusing to pair across frozen protocols")
    if left.get("source_hash") and right.get("source_hash") and left["source_hash"] != right["source_hash"]:
        raise StageError("refusing to pair across source identities")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        header = fields or ["status"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
        return
    header = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _pairing_signature(spec: dict[str, Any], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = spec.get("dataset") or {}
    training = spec.get("training") or {}
    opt = (training.get("optimizer") or {}) if isinstance(training.get("optimizer"), dict) else {}
    prefetch = spec.get("prefetch") or {}
    envelope = spec.get("resource_envelope") or {}
    model = spec.get("model") or {}
    summary = summary or {}
    return {
        "dataset": dataset.get("name"),
        "split_manifest": dataset.get("split_manifest"),
        "split_hash": spec.get("dataset_manifest_sha256") or summary.get("dataset_manifest_sha256"),
        "model_name": model.get("name"),
        "optimizer": opt.get("name"),
        "seed": (spec.get("seeds") or {}).get("model"),
        "eval_batch": training.get("eval_batch"),
        "precision": model.get("precision"),
        "quota_bytes": (spec.get("budget") or {}).get("limit_bytes"),
        "source_hash": spec.get("code_revision_or_snapshot") or summary.get("code_revision_or_snapshot"),
        "environment_hash": spec.get("environment_lock_sha256"),
        "frozen_hash": spec.get("frozen_hash"),
        "resource_sequence": json.dumps(envelope.get("reserved_bytes_by_experience") or [], separators=(",", ":")),
        "io_backend": (spec.get("data_supply") or {}).get("profile") or "default",
        "num_workers": prefetch.get("num_workers"),
        "pin_memory": prefetch.get("pin_memory"),
        "transform_fingerprint": json.dumps(dataset.get("transforms") or dataset.get("transform") or {}, sort_keys=True, default=str),
        "platform_gpu": None,
        "method": spec.get("method_id"),
        "new_batch": training.get("new_batch"),
        "replay_capacity": (spec.get("replay") or {}).get("capacity"),
        "plugin_policy": (spec.get("controller") or {}).get("plugin_policy"),
        "optional_plugins": (spec.get("algorithm") or {}).get("optional_plugins"),
        "latency_s": ((spec.get("controller") or {}).get("thresholds") or {}).get("latency_s"),
        "m_max_mib": ((spec.get("controller") or {}).get("thresholds") or {}).get("m_max_mib"),
    }


def _run_dir(context: StageContext, run_id: str | None) -> Path | None:
    if not run_id:
        return None
    path = context.root / "runs" / str(run_id)
    return path if path.is_dir() else None


def _load_spec(run_dir: Path | None, fallback: Path | None) -> dict[str, Any]:
    if run_dir and (run_dir / "resolved_config.yaml").is_file():
        return load_yaml(run_dir / "resolved_config.yaml")
    if fallback and fallback.is_file():
        return load_yaml(fallback)
    return {}


def _summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir and (run_dir / "summary.json").is_file():
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {}


def _mean_sd(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), None
    return float(mean(vals)), float(stdev(vals))


def _attempt_metric_row(item: dict[str, Any], attempt: dict[str, Any], context: StageContext) -> dict[str, Any]:
    cell = item.get("cell") or {}
    run_id = attempt.get("run_id")
    run_dir = _run_dir(context, run_id)
    spec_path = None
    cfg = cell.get("config") or attempt.get("config")
    if cfg:
        spec_path = context.root / cfg
    spec = _load_spec(run_dir, spec_path)
    summary = _summary(run_dir)
    sig = _pairing_signature(spec, summary)
    return {
        "cell_id": cell.get("cell_id") or attempt.get("cell_id"),
        "group": cell.get("group") or attempt.get("group"),
        "dataset": cell.get("dataset") or spec.get("dataset", {}).get("name") or attempt.get("dataset"),
        "method": cell.get("method") or spec.get("method_id") or attempt.get("method"),
        "seed": cell.get("seed") if cell.get("seed") is not None else sig.get("seed"),
        "variant": cell.get("variant") or attempt.get("variant"),
        "status": attempt.get("status") or summary.get("status") or "not_run",
        "run_id": run_id or "",
        "P": summary.get("p_diag"),
        "S": summary.get("s_initial"),
        "online_s": summary.get("online_total_s"),
        "failure_phase": summary.get("failure_phase") or attempt.get("failure_phase"),
        "n_trained": summary.get("n_experiences_trained"),
        "n_evaluated": summary.get("n_experiences_evaluated"),
        "elapsed_s": attempt.get("elapsed_s"),
        **{f"sig_{k}": v for k, v in sig.items()},
    }


def paired_diffs(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    by: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("group"), row.get("dataset"), row.get("seed"), row.get("sig_quota_bytes"), row.get("variant"))
        method = row.get("method")
        if method:
            by[key][str(method)] = row
    out = []
    allow = set(PAIR_ALLOW_BY_CONTRAST.get((left, right), set()))
    for key, methods in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if left not in methods or right not in methods:
            continue
        a, b = methods[left], methods[right]
        left_sig = {k[4:]: v for k, v in a.items() if str(k).startswith("sig_")}
        right_sig = {k[4:]: v for k, v in b.items() if str(k).startswith("sig_")}
        mismatch = None
        try:
            validate_pairs(left_sig, right_sig, allow=allow)
        except StageError as exc:
            mismatch = str(exc)
        item = {
            "group": key[0],
            "dataset": key[1],
            "seed": key[2],
            "quota_bytes": key[3],
            "variant": key[4],
            "left": left,
            "right": right,
            "left_run_id": a.get("run_id"),
            "right_run_id": b.get("run_id"),
            "left_status": a.get("status"),
            "right_status": b.get("status"),
            "pairing_ok": mismatch is None,
            "pairing_error": mismatch,
        }
        both = a.get("status") == "completed" and b.get("status") == "completed"
        item["both_completed"] = both
        for metric in ("P", "S", "online_s"):
            if not both or a.get(metric) in (None, "") or b.get(metric) in (None, ""):
                item[f"d_{metric}"] = None
            else:
                item[f"d_{metric}"] = float(a[metric]) - float(b[metric])
        out.append(item)
    return out


def _artifact_tables(bundle: dict[str, Any], context: StageContext) -> dict[str, list[dict[str, Any]]]:
    phase_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    plugin_rows: list[dict[str, Any]] = []
    supply_rows: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for item in bundle.get("cells") or []:
        for attempt in item.get("attempts") or []:
            run_id = attempt.get("run_id")
            if not run_id or run_id in seen_runs:
                continue
            seen_runs.add(run_id)
            run_dir = _run_dir(context, run_id)
            if run_dir is None:
                continue
            cell = item.get("cell") or {}
            base = {
                "run_id": run_id,
                "cell_id": cell.get("cell_id"),
                "method": cell.get("method"),
                "dataset": cell.get("dataset"),
                "seed": cell.get("seed"),
            }
            for row in _read_csv(run_dir / "phase_trace.csv"):
                phase_rows.append({**base, **row})
            for row in _read_jsonl(run_dir / "control_trace.jsonl"):
                control_rows.append({**base, **row})
            for row in _read_jsonl(run_dir / "plugin_activity.jsonl"):
                control_like = dict(row)
                if "optional_plugins" in control_like and not isinstance(control_like["optional_plugins"], str):
                    control_like["optional_plugins"] = json.dumps(control_like["optional_plugins"], default=str)
                plugin_rows.append({**base, **control_like})
            metrics = _read_csv(run_dir / "experience_metrics.csv")
            spec = _load_spec(run_dir, None)
            prefetch_on = bool((spec.get("prefetch") or {}).get("enabled"))
            for row in metrics:
                learning = float(row["learning_s"]) if row.get("learning_s") not in (None, "") else None
                produce = float(row.get("prefetch_produce_s") or 0.0)
                wait = float(row.get("prefetch_wait_s") or 0.0)
                blocked = wait if prefetch_on else produce
                supply_rows.append(
                    {
                        **base,
                        "experience_index": row.get("experience_index"),
                        "learning_s": learning,
                        "prefetch_produce_s": produce,
                        "prefetch_wait_s": wait,
                        "blocked_s": blocked,
                        "prefetch_enabled": prefetch_on,
                        "supply_wait_ratio": (blocked / learning) if learning else None,
                    }
                )
    return {
        "phase_resources": phase_rows,
        "control_events": control_rows,
        "plugin_activity": plugin_rows,
        "supply": supply_rows,
    }


def _search_cost_rows(context: StageContext) -> list[dict[str, Any]]:
    manifest_path = context.revision_dir / "probe_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest.get("runs") or []:
        if not str(row.get("role") or "").startswith("static_search"):
            continue
        rows.append(
            {
                "probe_id": row.get("probe_id"),
                "role": row.get("role"),
                "dataset": row.get("dataset"),
                "status": row.get("status"),
                "run_id": row.get("run_id"),
                "online_total_s": row.get("online_total_s"),
                "new_batch": row.get("new_batch"),
                "replay_capacity": row.get("replay_capacity"),
                "quota_mib": row.get("quota_mib"),
            }
        )
    return rows


def _means_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[(row.get("group"), row.get("dataset"), row.get("method"), row.get("variant"))].append(row)
    out = []
    for key, cells in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        completed = [
            c
            for c in cells
            if c.get("status") == "completed" and all(c.get(m) not in (None, "") for m in ("P", "S", "online_s"))
        ]
        p_m, p_sd = _mean_sd([float(c["P"]) for c in completed])
        s_m, s_sd = _mean_sd([float(c["S"]) for c in completed])
        t_m, t_sd = _mean_sd([float(c["online_s"]) for c in completed])
        out.append(
            {
                "group": key[0],
                "dataset": key[1],
                "method": key[2],
                "variant": key[3],
                "n_completed": len(completed),
                "n_attempts": len(cells),
                "n_failed": sum(1 for c in cells if c.get("status") in {"cuda_oom", "host_oom", "budget_exceeded", "implementation_error"}),
                "P_mean": p_m,
                "P_sd": p_sd,
                "S_mean": s_m,
                "S_sd": s_sd,
                "online_s_mean": t_m,
                "online_s_sd": t_sd,
                "completed_run_ids": ",".join(str(c.get("run_id") or "") for c in completed),
            }
        )
    return out


def _plot(metric_rows: list[dict[str, Any]], directory: Path) -> list[str]:
    completed = [r for r in metric_rows if r.get("status") == "completed" and r.get("P") not in (None, "") and r.get("S") not in (None, "")]
    if not completed:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = sorted({str(r.get("method")) for r in completed})
    for method in methods:
        xs = [float(r["S"]) for r in completed if str(r.get("method")) == method]
        ys = [float(r["P"]) for r in completed if str(r.get("method")) == method]
        ax.scatter(xs, ys, label=method)
    ax.set_xlabel("S_initial")
    ax.set_ylabel("P_diag")
    ax.set_title("effectiveness_v3 P vs S (not a verdict)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = directory / "fig_ps.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))
    timed = [r for r in completed if r.get("online_s") not in (None, "")]
    if timed:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = [[float(r["online_s"]) for r in timed if str(r.get("method")) == m] for m in methods]
        ax.boxplot(data, vert=True)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Online time (s)")
        ax.set_title("effectiveness_v3 online time (not a verdict)")
        fig.tight_layout()
        path = directory / "fig_online_s.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    return written


def render(bundle: dict[str, Any], context: StageContext, *, calibration: dict[str, Any] | None = None) -> Path:
    context.ensure_revision_dirs()
    coverage = audit_scenarios(calibration or {"study_id": STUDY_ID, "scenario_coverage": {}})
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for item in bundle.get("cells") or []:
        cell = item["cell"]
        if not item["attempts"]:
            coverage_rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "group": cell.get("group"),
                    "dataset": cell.get("dataset"),
                    "method": cell.get("method"),
                    "seed": cell.get("seed"),
                    "variant": cell.get("variant"),
                    "status": cell.get("exclude_reason") or "not_run",
                    "run_id": "",
                    "n_attempts": 0,
                }
            )
            attempt_rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "group": cell.get("group"),
                    "dataset": cell.get("dataset"),
                    "method": cell.get("method"),
                    "seed": cell.get("seed"),
                    "variant": cell.get("variant"),
                    "status": cell.get("exclude_reason") or "not_run",
                    "run_id": "",
                    "attempt_n": 0,
                }
            )
            continue
        coverage_rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "group": cell.get("group"),
                "dataset": cell.get("dataset"),
                "method": cell.get("method"),
                "seed": cell.get("seed"),
                "variant": cell.get("variant"),
                "status": item["attempts"][-1].get("status"),
                "run_id": item["attempts"][-1].get("run_id") or "",
                "n_attempts": item["n_attempts"],
                "attempt_run_ids": ",".join(str(a.get("run_id") or "") for a in item["attempts"]),
            }
        )
        for i, attempt in enumerate(item["attempts"]):
            attempt_rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "group": cell.get("group"),
                    "dataset": cell.get("dataset"),
                    "method": cell.get("method"),
                    "seed": cell.get("seed"),
                    "variant": cell.get("variant"),
                    "status": attempt.get("status"),
                    "run_id": attempt.get("run_id") or "",
                    "attempt_n": i + 1,
                    "elapsed_s": attempt.get("elapsed_s"),
                }
            )
            metric_rows.append(_attempt_metric_row(item, attempt, context))
    artifacts = _artifact_tables(bundle, context)
    means = _means_rows(metric_rows)
    paired = (
        paired_diffs(metric_rows, "O11", "S0")
        + paired_diffs(metric_rows, "O11", "S_star")
        + paired_diffs(metric_rows, "O11", "R11")
        + paired_diffs(metric_rows, "O11", "F11")
        + paired_diffs(metric_rows, "O11", "O00")
        + paired_diffs(metric_rows, "O11", "O10")
        + paired_diffs(metric_rows, "O11", "O01")
    )
    _write_csv(context.report_dir / "coverage.csv", coverage_rows)
    _write_csv(context.report_dir / "attempts.csv", attempt_rows)
    _write_csv(context.report_dir / "means.csv", means)
    _write_csv(context.report_dir / "paired.csv", paired)
    _write_csv(context.report_dir / "phase_resources.csv", artifacts["phase_resources"])
    _write_csv(
        context.report_dir / "control_events.csv",
        [{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in artifacts["control_events"]],
    )
    _write_csv(context.report_dir / "plugin_activity.csv", artifacts["plugin_activity"])
    _write_csv(context.report_dir / "supply.csv", artifacts["supply"])
    _write_csv(context.report_dir / "search_cost.csv", _search_cost_rows(context))
    atomic_write_json(context.report_dir / "scenario_coverage.json", coverage)
    selection = (calibration or {}).get("datasets") or {}
    atomic_write_json(
        context.report_dir / "selection.json",
        {
            "study_id": STUDY_ID,
            "revision": context.revision,
            "static_selections": {name: (block or {}).get("static_selections") for name, block in selection.items()},
            "note": "Search cost is in search_cost.csv; selected S* is not a runtime speedup",
        },
    )
    figures = _plot(metric_rows, context.report_dir / "figures")
    n_eligible = bundle.get("eligible_n")
    n_design = bundle.get("design_slots")
    lines = [
        f"# effectiveness_v3 {context.revision}",
        "",
        f"design_slots={n_design} eligible={n_eligible} excluded={bundle.get('excluded_n')}",
        "",
        "判定尚未用正式结果填写。失败与未建立场景保留。配对表保留签名失败行，不挑最快 attempt。",
        "测试通过、URGE 触发次数或完成格数都不是有效性成立。",
        "",
        f"tables={', '.join(TABLE_NAMES)}",
        f"figures={len(figures)}",
        "",
    ]
    for scenario_id, item in (coverage.get("scenarios") or {}).items():
        lines.append(
            f"- {scenario_id}: construction={item.get('construction_status')} verdict={item.get('effectiveness_verdict')}"
        )
    results = context.report_dir / "RESULTS.md"
    context.assert_output_path(results)
    atomic_write_text(results, "\n".join(lines) + "\n")
    return results


def render_development(manifest: dict[str, Any], context: StageContext) -> Path:
    """G2 tables from probe_manifest. Not a formal C01–C08 report."""
    context.ensure_revision_dirs()
    cells = []
    for row in manifest.get("runs") or []:
        cell_id = f"dev/{row.get('dataset')}/{row.get('role')}/{row.get('probe_id')}/0"
        cells.append(
            {
                "cell": {
                    "cell_id": cell_id,
                    "group": row.get("role"),
                    "dataset": row.get("dataset"),
                    "method": row.get("config_id") or row.get("probe_id"),
                    "seed": 0,
                    "variant": row.get("role"),
                    "config": row.get("config"),
                },
                "attempts": [
                    {
                        "status": row.get("status"),
                        "run_id": row.get("run_id"),
                        "cell_id": cell_id,
                        "config": row.get("config"),
                    }
                ]
                if row.get("status") not in {None, "not_run"}
                else [],
                "n_attempts": 1 if row.get("run_id") else 0,
            }
        )
    bundle = {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "design_slots": None,
        "eligible_n": None,
        "excluded_n": None,
        "cells": cells,
        "attempts": [c["attempts"][0] for c in cells if c["attempts"]],
    }
    return render(bundle, context, calibration={"study_id": STUDY_ID, "scenario_coverage": {}})
