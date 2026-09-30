"""Isolated log-disk-full simulation. Does not fill the system root."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from orion_repro.stages.fullmem_v4.constants import MIN_FREE_BYTES, ORION_PYTHON, STUDY_ID
from orion_repro.stages.fullmem_v4.executor import process_one
from orion_repro.stages.fullmem_v4.queue import list_inbox, load_state, stage_paths, submit
from orion_repro.stages.fullmem_v4.util import atomic_write_json, utc_now


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    report_path = Path(sys.argv[2]).resolve()
    usage = shutil.disk_usage(root)
    inbox_before = list_inbox(root)
    submitted = submit(
        {
            "study_id": STUDY_ID,
            "task_id": "g1-diskfull-a",
            "seq": 810,
            "kind": "dummy",
            "argv": [
                ORION_PYTHON,
                "-m",
                "orion_repro.stages.fullmem_v4",
                "dummy",
                "--seconds",
                "1",
                "--marker",
                "runs/fullmem_v4/g1-diskfull-a.ok",
            ],
            "required_capacity": "any",
            "batch_id": "g1-diskfull",
            "working_directory": str(root),
        },
        root=root,
    )
    identity = {"boot_id": "diskfull-probe", "capacity": {"capacity_id": "any"}}
    inbox = list_inbox(root)
    status = process_one(inbox[0], load_state(root), identity, root)
    state = load_state(root)
    paths = stage_paths(root)
    payload = {
        "ts_utc": utc_now(),
        "root": str(root),
        "fstype_df_free_bytes": int(usage.free),
        "fstype_df_total_bytes": int(usage.total),
        "min_free_bytes": MIN_FREE_BYTES,
        "below_threshold": bool(usage.free < MIN_FREE_BYTES),
        "submitted": submitted,
        "inbox_before_n": len(inbox_before),
        "process_one": status,
        "mode": state.get("mode"),
        "pause_reason": state.get("pause_reason"),
        "current_task_id": state.get("current_task_id"),
        "inbox_kept": (paths["inbox"] / "g1-diskfull-a.json").is_file(),
        "running_absent": not (paths["running"] / "g1-diskfull-a.json").exists(),
        "done_absent": not (paths["done"] / "g1-diskfull-a.json").exists(),
        "host_pid": os.getpid(),
    }
    print(json.dumps(payload, indent=2))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, payload)
    if not payload["below_threshold"]:
        print("FAIL: probe root is not below MIN_FREE_BYTES", file=sys.stderr)
        return 2
    if status != "paused" or not str(state.get("pause_reason") or "").startswith("disk_low:"):
        print("FAIL: expected disk_low pause", file=sys.stderr)
        return 3
    if not payload["inbox_kept"] or not payload["running_absent"]:
        print("FAIL: inbox was consumed", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
