"""Classify formal run outcomes. Does not drop failures or fill zeros."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "run_id",
    "status",
    "phase",
    "dataset",
    "method_id",
    "algorithm_base",
    "seed",
    "protocol_id",
    "timing_schema",
    "n_experiences_run",
    "P_diag",
    "S_initial",
    "online_total_s",
    "error_kind",
]


def _spec(run_dir: Path) -> dict:
    yml = run_dir / "resolved_config.yaml"
    if yml.is_file():
        return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    js = run_dir / "resolved_config.yaml.json"
    if js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    return {}


def _seed(spec: dict, summary: dict):
    seeds = spec.get("seeds")
    if isinstance(seeds, dict) and seeds.get("model") is not None:
        return int(seeds["model"])
    if summary.get("seed") is not None:
        return int(summary["seed"])
    return None


def error_kind(summary: dict) -> str:
    status = summary.get("status") or ""
    if status == "completed":
        p = summary.get("p_diag")
        if p is not None and float(p) < 0.05:
            return "completed_collapse"
        return "ok"
    tb = summary.get("traceback") or summary.get("reason") or ""
    blob = str(tb).lower()
    if status == "cuda_oom" or "out of memory" in blob:
        return "cuda_oom"
    if status == "budget_exceeded":
        return "budget_exceeded"
    if "multinomial" in blob:
        return "gss_multinomial"
    if "empty sequence" in blob:
        return "gss_empty_mem_grads"
    if "has no setter" in blob:
        return "naive_mb_x_setter"
    if status == "implementation_error":
        return "implementation_error"
    if status == "running" or not status:
        return "incomplete"
    return status


def main() -> None:
    rows = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("phase") != "formal":
            continue
        spec = _spec(path.parent)
        if spec.get("study_id") == "light24_v1":
            continue
        rows.append(
            {
                "run_id": summary.get("run_id") or path.parent.name,
                "status": summary.get("status"),
                "phase": summary.get("phase"),
                "dataset": spec.get("dataset", {}).get("name") or summary.get("dataset"),
                "method_id": spec.get("method_id") or summary.get("method_id"),
                "algorithm_base": (spec.get("algorithm") or {}).get("base"),
                "seed": _seed(spec, summary),
                "protocol_id": spec.get("protocol_id") or summary.get("protocol_id"),
                "timing_schema": summary.get("timing_schema"),
                "n_experiences_run": summary.get("n_experiences_run"),
                "P_diag": summary.get("p_diag"),
                "S_initial": summary.get("s_initial"),
                "online_total_s": summary.get("online_total_s"),
                "error_kind": error_kind(summary),
            }
        )
    out = ROOT / "reports" / "formal_failures.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts = Counter((r["status"], r["error_kind"]) for r in rows)
    print(json.dumps({"n": len(rows), "counts": {str(k): v for k, v in counts.items()}, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
