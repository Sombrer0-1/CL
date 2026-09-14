"""Rebuild reports/results.csv from run summaries.

Does not drop failures. usage is a conservative label, not a claim that
the row is a frozen formal result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
    "run_id",
    "phase",
    "dataset",
    "method_id",
    "status",
    "protocol_id",
    "seed",
    "experiences",
    "P_diag",
    "S_initial",
    "online_total_s",
    "timing_schema",
    "usage",
]


def classify_usage(summary: dict) -> str:
    phase = summary.get("phase")
    timing = summary.get("timing_schema")
    if phase == "formal":
        return "formal"
    if timing == "online_loop_v2":
        return "current_code_development"
    if timing == "legacy_includes_setup":
        return "legacy_includes_setup"
    return "unclassified"


def iter_summaries(runs_dir: Path):
    for path in sorted(runs_dir.glob("*/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            yield payload


def row_from_summary(summary: dict) -> dict:
    return {
        "run_id": summary.get("run_id"),
        "phase": summary.get("phase"),
        "dataset": summary.get("dataset"),
        "method_id": summary.get("method_id"),
        "status": summary.get("status"),
        "protocol_id": summary.get("protocol_id") or summary.get("eval_protocol"),
        "seed": summary.get("seed", summary.get("seeds", {}).get("model") if isinstance(summary.get("seeds"), dict) else None),
        "experiences": summary.get("n_experiences_run"),
        "P_diag": summary.get("p_diag"),
        "S_initial": summary.get("s_initial"),
        "online_total_s": summary.get("online_total_s"),
        "timing_schema": summary.get("timing_schema"),
        "usage": classify_usage(summary),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "results.csv"))
    args = parser.parse_args(argv)
    rows = [row_from_summary(s) for s in iter_summaries(Path(args.runs))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
