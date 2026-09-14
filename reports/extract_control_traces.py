"""Summarize URGE control traces from completed formal runs.

Does not train. Walks runs/*/summary.json, reads control_trace.jsonl and
resolved_config.yaml, writes reports/formal_control_summary.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "run_id",
    "dataset",
    "algorithm_base",
    "method_id",
    "seed",
    "n_records",
    "first_applied_batch",
    "last_applied_batch",
    "first_applied_replay",
    "last_applied_replay",
    "n_advanced",
    "n_default",
    "n_urge_gt_thr",
    "max_latency_s",
    "min_urge",
    "max_urge",
]


def _spec(run_dir: Path) -> dict:
    yml = run_dir / "resolved_config.yaml"
    if yml.is_file():
        return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    js = run_dir / "resolved_config.yaml.json"
    if js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    return {}


def _seed(spec: dict, summary: dict) -> int | None:
    seeds = spec.get("seeds")
    if isinstance(seeds, dict) and seeds.get("model") is not None:
        return int(seeds["model"])
    if summary.get("seed") is not None:
        return int(summary["seed"])
    nested = summary.get("seeds")
    if isinstance(nested, dict) and nested.get("model") is not None:
        return int(nested["model"])
    return None


def _load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def summarize_trace(records: list[dict]) -> dict:
    n_advanced = 0
    n_default = 0
    n_urge_gt_thr = 0
    first_batch = None
    last_batch = None
    first_replay = None
    last_replay = None
    latencies: list[float] = []
    urges: list[float] = []
    for row in records:
        if "applied_new_batch" in row and row["applied_new_batch"] is not None:
            batch = int(row["applied_new_batch"])
            if first_batch is None:
                first_batch = batch
            last_batch = batch
        if "applied_replay" in row and row["applied_replay"] is not None:
            replay = int(row["applied_replay"])
            if first_replay is None:
                first_replay = replay
            last_replay = replay
        mode = row.get("applied_optimizer_mode")
        if mode == "advanced":
            n_advanced += 1
        elif mode == "default":
            n_default += 1
        urge = row.get("urge")
        thr = row.get("thr")
        if urge is not None:
            urges.append(float(urge))
            if thr is not None and float(urge) > float(thr):
                n_urge_gt_thr += 1
        latency = row.get("latency_s")
        if latency is not None:
            latencies.append(float(latency))
    return {
        "n_records": len(records),
        "first_applied_batch": first_batch,
        "last_applied_batch": last_batch,
        "first_applied_replay": first_replay,
        "last_applied_replay": last_replay,
        "n_advanced": n_advanced,
        "n_default": n_default,
        "n_urge_gt_thr": n_urge_gt_thr,
        "max_latency_s": max(latencies) if latencies else None,
        "min_urge": min(urges) if urges else None,
        "max_urge": max(urges) if urges else None,
    }


def main() -> None:
    rows: list[dict] = []
    skipped_no_trace = 0
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("phase") != "formal" or summary.get("status") != "completed":
            continue
        run_dir = path.parent
        ctrl = run_dir / "control_trace.jsonl"
        if not ctrl.is_file():
            skipped_no_trace += 1
            continue
        spec = _spec(run_dir)
        if spec.get("study_id") == "light24_v1":
            continue
        recs = _load_records(ctrl)
        trace = summarize_trace(recs)
        rows.append(
            {
                "run_id": summary.get("run_id") or run_dir.name,
                "dataset": (spec.get("dataset") or {}).get("name") or summary.get("dataset"),
                "algorithm_base": (spec.get("algorithm") or {}).get("base"),
                "method_id": spec.get("method_id") or summary.get("method_id"),
                "seed": _seed(spec, summary),
                **trace,
            }
        )
    out = ROOT / "reports" / "formal_control_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_orion = sum(1 for r in rows if r.get("method_id") == "orion_formula")
    n_adapted = sum(
        1
        for r in rows
        if r.get("method_id") == "orion_formula"
        and (
            (
                r.get("first_applied_batch") is not None
                and r.get("last_applied_batch") is not None
                and r["last_applied_batch"] != r["first_applied_batch"]
            )
            or int(r.get("n_advanced") or 0) > 0
        )
    )
    print(
        json.dumps(
            {
                "n": len(rows),
                "skipped_no_trace": skipped_no_trace,
                "out": str(out),
                "n_orion_formula": n_orion,
                "n_orion_adapted_batch_or_advanced": n_adapted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
