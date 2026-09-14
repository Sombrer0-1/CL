"""Tabulate formal completed runs from summaries. Failures are kept."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="formal")
    parser.add_argument("--out", default=str(ROOT / "reports" / "formal_main_table.csv"))
    args = parser.parse_args(argv)
    rows = []
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("phase") != args.phase:
            continue
        rows.append(
            {
                "run_id": s.get("run_id"),
                "status": s.get("status"),
                "dataset": s.get("dataset"),
                "method_id": s.get("method_id"),
                "protocol_id": s.get("protocol_id"),
                "n_experiences": s.get("n_experiences_run"),
                "p_diag": s.get("p_diag"),
                "s_initial": s.get("s_initial"),
                "online_total_s": s.get("online_total_s"),
                "timing_schema": s.get("timing_schema"),
                "reuse_identity": s.get("reuse_identity"),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "run_id", "status", "dataset", "method_id", "p_diag", "s_initial", "online_total_s"
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        if r.get("status") == "completed" and r.get("p_diag") is not None:
            groups[(r["dataset"], r["method_id"])].append(r)
    means = []
    for key, items in sorted(groups.items()):
        import statistics as st
        ps = [float(x["p_diag"]) for x in items]
        ss = [float(x["s_initial"]) for x in items]
        ts = [float(x["online_total_s"]) for x in items]
        means.append(
            {
                "dataset": key[0],
                "method_id": key[1],
                "n": len(items),
                "p_mean": st.mean(ps),
                "p_std": st.pstdev(ps) if len(ps) > 1 else 0.0,
                "s_mean": st.mean(ss),
                "s_std": st.pstdev(ss) if len(ss) > 1 else 0.0,
                "t_mean": st.mean(ts),
                "t_std": st.pstdev(ts) if len(ts) > 1 else 0.0,
            }
        )
    mean_path = out.with_name("formal_means.csv")
    if means:
        with mean_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(means[0].keys()))
            w.writeheader()
            w.writerows(means)
    print(json.dumps({"n_rows": len(rows), "n_groups": len(means), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
