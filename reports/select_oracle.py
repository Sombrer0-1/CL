"""Apply frozen A15 Oracle selection to completed development cells.

Ignores legacy_includes_setup and missing timing_schema. Does not train.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def learning_sum(run_dir: Path) -> float | None:
    path = run_dir / "experience_metrics.csv"
    if not path.is_file():
        return None
    total = 0.0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += float(row.get("learning_s") or 0.0)
    return total


def peak_device(run_dir: Path) -> float | None:
    path = run_dir / "experience_metrics.csv"
    if not path.is_file():
        return None
    peak = 0.0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            peak = max(peak, float(row.get("memory_mib") or 0.0))
    return peak


def dominates(a: dict, b: dict) -> bool:
    """True if a is strictly better on all maximized/minimized objectives, or
    weakly better on all and strictly better on at least one.
    Maximize P,S; minimize learning_s.
    """
    better_or_equal = (
        a["p_diag"] >= b["p_diag"]
        and a["s_initial"] >= b["s_initial"]
        and a["learning_s"] <= b["learning_s"]
    )
    strictly = (
        a["p_diag"] > b["p_diag"]
        or a["s_initial"] > b["s_initial"]
        or a["learning_s"] < b["learning_s"]
    )
    return better_or_equal and strictly


def pareto_front(rows: list[dict]) -> list[dict]:
    front = []
    for cand in rows:
        if any(dominates(other, cand) for other in rows if other is not cand):
            continue
        front.append(cand)
    return front


def collect_cells(runs: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs.glob("*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if s.get("status") != "completed":
            continue
        if s.get("method_id") != "oracle_reconstructed":
            continue
        if s.get("timing_schema") != "online_loop_v2":
            continue
        if s.get("phase") != "development":
            continue
        run_dir = summary_path.parent
        cfg = run_dir / "resolved_config.yaml.json"
        cell_id = None
        dataset = s.get("dataset")
        if cfg.is_file():
            spec = json.loads(cfg.read_text(encoding="utf-8"))
            cell_id = spec.get("oracle_cell_id")
            dataset = spec.get("dataset", {}).get("name", dataset)
        learn = learning_sum(run_dir)
        rows.append(
            {
                "run_id": s.get("run_id"),
                "dataset": dataset,
                "oracle_cell_id": cell_id or "",
                "p_diag": float(s.get("p_diag")),
                "s_initial": float(s.get("s_initial")),
                "learning_s": float(learn if learn is not None else s.get("online_total_s") or 0.0),
                "peak_device_mib": peak_device(run_dir),
                "online_total_s": float(s.get("online_total_s") or 0.0),
            }
        )
    return rows


def select_group(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "selected": None, "pareto": [], "p_best": None, "s_best": None}
    front = pareto_front(rows)
    def key(r):
        score = 0.5 * r["p_diag"] + 0.5 * r["s_initial"]
        peak = r["peak_device_mib"] if r["peak_device_mib"] is not None else float("inf")
        return (-score, r["learning_s"], peak, r["oracle_cell_id"])
    selected = sorted(front, key=key)[0]
    p_best = max(rows, key=lambda r: (r["p_diag"], -r["learning_s"]))
    s_best = max(rows, key=lambda r: (r["s_initial"], -r["learning_s"]))
    return {
        "n": len(rows),
        "selected": selected,
        "pareto": sorted(front, key=key),
        "p_best": p_best,
        "s_best": s_best,
        "rule": "A15: Pareto then max 0.5P+0.5S, then shorter learning_s, then smaller peak, then cell_id",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "oracle_a15.json"))
    args = parser.parse_args()
    cells = collect_cells(Path(args.runs))
    by_ds: dict[str, list] = {}
    for row in cells:
        by_ds.setdefault(str(row["dataset"]), []).append(row)
    payload = {ds: select_group(group) for ds, group in sorted(by_ds.items())}
    payload["_meta"] = {
        "n_eligible_cells": len(cells),
        "note": "Only development oracle_reconstructed with online_loop_v2. Legacy grids are excluded.",
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_eligible": len(cells), "datasets": list(by_ds)}, indent=2))


if __name__ == "__main__":
    main()
