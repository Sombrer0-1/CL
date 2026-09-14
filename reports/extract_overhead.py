"""E09/C08 overhead from existing runs: controller + reconfigure vs learning.

Does not train. Energy is out of platform scope and is not inferred here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _spec(run_dir: Path) -> dict:
    yml = run_dir / "resolved_config.yaml"
    if yml.is_file():
        return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    js = run_dir / "resolved_config.yaml.json"
    if js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    return {}


def control_overhead(run_dir: Path) -> dict:
    ctrl = run_dir / "control_trace.jsonl"
    metrics = run_dir / "experience_metrics.csv"
    controller_s = 0.0
    reconfigure_s = 0.0
    n_ctrl = 0
    if ctrl.is_file():
        for line in ctrl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            controller_s += float(row.get("controller_s") or 0.0)
            reconfigure_s += float(row.get("reconfigure_s") or 0.0)
            n_ctrl += 1
    learning_s = 0.0
    evaluation_s = 0.0
    prefetch_wait_s = 0.0
    if metrics.is_file():
        with metrics.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                learning_s += float(row.get("learning_s") or 0.0)
                evaluation_s += float(row.get("evaluation_s") or 0.0)
                prefetch_wait_s += float(row.get("prefetch_wait_s") or 0.0)
    denom = learning_s if learning_s > 0 else None
    control_s = controller_s + reconfigure_s
    return {
        "controller_s": controller_s,
        "reconfigure_s": reconfigure_s,
        "control_s": control_s,
        "n_control_records": n_ctrl,
        "learning_s": learning_s,
        "evaluation_s": evaluation_s,
        "prefetch_wait_s": prefetch_wait_s,
        "control_over_learning_pct": (100.0 * control_s / denom) if denom else None,
        "prefetch_over_learning_pct": (100.0 * prefetch_wait_s / denom) if denom else None,
    }


def main() -> None:
    rows = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("phase") != "formal" or s.get("status") != "completed":
            continue
        spec = _spec(path.parent)
        if spec.get("study_id") == "light24_v1":
            continue
        oh = control_overhead(path.parent)
        rows.append(
            {
                "run_id": s.get("run_id"),
                "dataset": spec.get("dataset", {}).get("name") or s.get("dataset"),
                "algorithm_base": (spec.get("algorithm") or {}).get("base"),
                "method_id": spec.get("method_id") or s.get("method_id"),
                "online_total_s": s.get("online_total_s"),
                **oh,
            }
        )
        online = float(s.get("online_total_s") or 0.0)
        control_s = float(oh["control_s"])
        rows[-1]["control_over_online_pct"] = (
            (100.0 * control_s / online) if online > 0 else None
        )
    out = ROOT / "reports" / "formal_overhead.csv"
    if rows:
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({"n": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
