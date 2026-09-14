"""Tabulate formal runs by dataset + algorithm.base + method_id.

Does not train. Reads resolved_config so GEM/AGEM/GSS are not mixed into ER.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_spec(run_dir: Path) -> dict:
    js = run_dir / "resolved_config.yaml.json"
    if js.is_file():
        return json.loads(js.read_text(encoding="utf-8"))
    yml = run_dir / "resolved_config.yaml"
    if yml.is_file():
        return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    return {}


def collect(phase: str = "formal") -> list[dict]:
    rows = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("phase") != phase:
            continue
        spec = _load_spec(path.parent)
        if spec.get("study_id") == "light24_v1":
            continue
        algo = spec.get("algorithm") or {}
        ctrl = spec.get("controller") or {}
        rows.append(
            {
                "run_id": s.get("run_id"),
                "status": s.get("status"),
                "dataset": spec.get("dataset", {}).get("name") or s.get("dataset"),
                "algorithm_base": algo.get("base"),
                "optional_plugins": algo.get("optional_plugins"),
                "method_id": spec.get("method_id") or s.get("method_id"),
                "preference": ctrl.get("preference") or "",
                "preference_order": json.dumps(ctrl.get("preference_order") or []),
                "experiment_id": spec.get("experiment_id") or s.get("experiment_id"),
                "protocol_id": spec.get("protocol_id") or s.get("protocol_id"),
                "n_experiences": s.get("n_experiences_run"),
                "p_diag": s.get("p_diag"),
                "s_initial": s.get("s_initial"),
                "online_total_s": s.get("online_total_s"),
                "timing_schema": s.get("timing_schema"),
                "failure": s.get("failure_reason") or s.get("error") or "",
            }
        )
    return rows


def means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r.get("status") == "completed" and r.get("p_diag") is not None:
            key = (r["dataset"], r["algorithm_base"], r["method_id"], r["preference"])
            groups[key].append(r)
    out = []
    for key, items in sorted(groups.items()):
        ps = [float(x["p_diag"]) for x in items]
        ss = [float(x["s_initial"]) for x in items]
        ts = [float(x["online_total_s"]) for x in items]
        out.append(
            {
                "dataset": key[0],
                "algorithm_base": key[1],
                "method_id": key[2],
                "preference": key[3],
                "n": len(items),
                "n_planned_note": "n is completed with P; failures live in formal_context_table.csv",
                "p_mean": st.mean(ps),
                "p_std": st.pstdev(ps) if len(ps) > 1 else None,
                "s_mean": st.mean(ss),
                "s_std": st.pstdev(ss) if len(ss) > 1 else None,
                "t_mean": st.mean(ts),
                "t_std": st.pstdev(ts) if len(ts) > 1 else None,
            }
        )
    return out


def main() -> None:
    rows = collect()
    table = ROOT / "reports" / "formal_context_table.csv"
    mean_path = ROOT / "reports" / "formal_context_means.csv"
    fields = list(rows[0].keys()) if rows else ["run_id", "status"]
    with table.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    m = means(rows)
    if m:
        with mean_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(m[0].keys()))
            w.writeheader()
            w.writerows(m)
    print(
        json.dumps(
            {"n_rows": len(rows), "n_groups": len(m), "table": str(table), "means": str(mean_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
