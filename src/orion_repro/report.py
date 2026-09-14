"""Summarize the latest state of each run, keeping event counts separate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize_registry(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {}
    for event in events:
        run_id = event["run_id"]
        latest[run_id] = {**latest.get(run_id, {}), **event}
    rows = list(latest.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(path),
        "n_records": len(events),
        "n_runs": len(rows),
        "event_status_counts": dict(Counter(r.get("status") for r in events)),
        "status_counts": dict(Counter(r.get("status") for r in rows)),
        "completed_runs": [r for r in rows if r.get("status") == "completed"],
        "failed_runs": [r for r in rows if r.get("status") not in {None, "running", "completed", "planned"}],
        "pending_runs": [r for r in rows if r.get("status") in {"running", "planned"}],
        "note": "status_counts counts latest registry state per run_id, not events. Historical events are retained. Unrun matrix cells are omitted, not zero. Registry-only reconciliations are not original training summaries. This is an inventory, not a formal statistical report.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(ROOT / "experiments" / "registry.jsonl"))
    args = parser.parse_args(argv)
    path = Path(args.registry)
    payload = summarize_registry(path)
    out = ROOT / "reports" / "registry_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
