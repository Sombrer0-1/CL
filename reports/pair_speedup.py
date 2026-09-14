"""Paired Orion vs baseline speedup from completed formal runs. Not a C03 verdict."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _spec(run_dir: Path) -> dict:
    yml = run_dir / "resolved_config.yaml"
    return yaml.safe_load(yml.read_text(encoding="utf-8")) if yml.is_file() else {}


def collect() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("phase") != "formal" or s.get("status") != "completed":
            continue
        if s.get("timing_schema") != "online_loop_v2":
            continue
        spec = _spec(path.parent)
        if spec.get("study_id") == "light24_v1":
            continue
        rows.append(
            {
                "dataset": spec.get("dataset", {}).get("name") or s.get("dataset"),
                "algorithm_base": (spec.get("algorithm") or {}).get("base"),
                "method_id": spec.get("method_id"),
                "seed": (spec.get("seeds") or {}).get("model"),
                "p_diag": s.get("p_diag"),
                "s_initial": s.get("s_initial"),
                "online_total_s": s.get("online_total_s"),
                "run_id": s.get("run_id"),
            }
        )
    return rows


def main() -> None:
    rows = collect()
    by_key = defaultdict(dict)
    for r in rows:
        if r["p_diag"] is None or r["seed"] is None:
            continue
        key = (r["dataset"], r["algorithm_base"], int(r["seed"]))
        by_key[key][r["method_id"]] = r
    out_rows = []
    for key, methods in sorted(by_key.items()):
        orion = methods.get("orion_formula")
        if orion is None:
            continue
        for baseline in ("er_static", "max_a_reconstructed", "max_p_reconstructed", "lr_reconstructed"):
            other = methods.get(baseline)
            if other is None:
                continue
            t_o = float(orion["online_total_s"])
            t_b = float(other["online_total_s"])
            out_rows.append(
                {
                    "dataset": key[0],
                    "algorithm_base": key[1],
                    "seed": key[2],
                    "baseline": baseline,
                    "orion_p": orion["p_diag"],
                    "baseline_p": other["p_diag"],
                    "orion_s": orion["s_initial"],
                    "baseline_s": other["s_initial"],
                    "orion_t": t_o,
                    "baseline_t": t_b,
                    "speedup_t": (t_b / t_o) if t_o > 0 else None,
                    "delta_p": float(orion["p_diag"]) - float(other["p_diag"]),
                    "orion_run": orion["run_id"],
                    "baseline_run": other["run_id"],
                }
            )
    path = ROOT / "reports" / "formal_pair_speedup.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows else ["dataset"])
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps({"n_pairs": len(out_rows), "out": str(path)}, indent=2))


if __name__ == "__main__":
    main()
