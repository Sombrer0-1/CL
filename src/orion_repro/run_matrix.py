"""Sequential matrix execution (PLAN §12, §8.6). Fresh process per config."""

from __future__ import annotations

import argparse
import json
import subprocess
import re
import time
import sys
from pathlib import Path

from orion_repro.runner.spec import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def parse_child_summary(stdout: str) -> dict | None:
    # Logs may precede a pretty-printed JSON object. Read whole objects, not lines.
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"(?m)^\s*\{", stdout))):
        try:
            value, _ = decoder.raw_decode(stdout[match.start():].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "run_id" in value and "status" in value:
            return value
    return None


def _run_one(path: Path) -> dict:
    cmd = [sys.executable, "-m", "orion_repro.run", "--config", str(path)]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=None,
    )
    stdout = proc.stdout[-8000:] if proc.stdout else ""
    stderr = proc.stderr[-4000:] if proc.stderr else ""
    summary = {
        "config": str(path),
        "process_wall_s": time.monotonic() - started,
        "returncode": proc.returncode,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }
    parsed = parse_child_summary(proc.stdout or "")
    if isinstance(parsed, dict):
        summary.update(parsed)
    if proc.returncode != 0 and summary.get("status") not in {
        "cuda_oom",
        "host_oom",
        "budget_exceeded",
        "preflight_infeasible",
    }:
        summary["status"] = "implementation_error"
    elif proc.returncode == 0:
        summary.setdefault("status", "completed")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse completed identical runs with frozen manifest; restart incomplete runs")
    args = parser.parse_args(argv)
    matrix = load_yaml(Path(args.matrix))
    configs = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in matrix.get("configs", [])]
    results = []
    for path in configs:
        from orion_repro.runner.reuse import find_completed, reuse_identity
        previous = find_completed(ROOT, reuse_identity(load_yaml(path), ROOT)) if args.resume else None
        if previous is not None:
            results.append({"config": str(path), "status": "reused", "run_id": previous["run_id"]})
            continue
        summary = _run_one(path)
        results.append(summary)
        status = summary.get("status")
        if status == "implementation_error" or (
            summary.get("returncode", 0) not in {0} and status not in {"cuda_oom", "host_oom", "budget_exceeded"}
        ):
            print(json.dumps({"paused": True, "reason": "implementation_error", "config": str(path), "summary": summary}, default=str))
            break
    print(json.dumps({"n": len(results), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
