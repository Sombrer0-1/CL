"""Calibrate H4 background hold bytes on the current admitted capacity.

Stops before exhausting the board. Writes development hold amounts; not a G3 freeze.
"""

from __future__ import annotations

import gc
import time
from typing import Any

from orion_repro.memory.h4_holder import BackgroundHolder, H4HolderError, memavailable_bytes, meminfo_kb
from orion_repro.stages.fullmem_v4.constants import ORION_PYTHON, STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now

CHUNK_BYTES = 256 * 1024 * 1024
# Leave OS/SSH/CUDA room during idle calibration. Training may go lower.
SAFETY_FLOOR_BYTES = 1024 * 1024 * 1024
# High vs idle must drop at least this much to count as identifiable.
MIN_IDENTIFIABLE_DROP_BYTES = 1024 * 1024 * 1024
MAX_HOLD_BYTES = 4 * 1024 * 1024 * 1024


def calibrate_h4_hold() -> dict[str, Any]:
    identity = collect_identity(include_cuda=False)
    cap = (identity.get("capacity") or {}).get("capacity_id")
    baseline = memavailable_bytes()
    steps: list[dict[str, Any]] = []
    chosen = 0
    recoverable = False
    holder = BackgroundHolder.start(executable=ORION_PYTHON)
    try:
        held = 0
        while held + CHUNK_BYTES <= MAX_HOLD_BYTES:
            nxt = held + CHUNK_BYTES
            avail_now = memavailable_bytes()
            if avail_now - CHUNK_BYTES < SAFETY_FLOOR_BYTES:
                steps.append(
                    {
                        "held_bytes": held,
                        "next_would_be": nxt,
                        "mem_available_bytes": avail_now,
                        "stop": "safety_floor",
                    }
                )
                break
            rec = holder.transition(len(steps), nxt)
            time.sleep(0.4)
            avail = memavailable_bytes()
            row = {
                "held_bytes": nxt,
                "actual": rec.actual_tensor_bytes,
                "status": rec.status,
                "mem_available_bytes": avail,
                "drop_from_baseline_bytes": baseline - avail,
                "holder_rss_kb": rec.holder_rss_kb,
                "pid": rec.holder_pid,
            }
            steps.append(row)
            if rec.status != "ok" or rec.actual_tensor_bytes != nxt:
                break
            if avail < SAFETY_FLOOR_BYTES:
                row["stop"] = "crossed_floor"
                break
            held = nxt
            if baseline - avail >= MIN_IDENTIFIABLE_DROP_BYTES:
                chosen = held
        if chosen == 0 and held > 0:
            # Largest safe hold even if drop was short of 1GiB.
            chosen = held
        holder.transition(len(steps), 0)
        time.sleep(0.8)
        gc.collect()
        recovered = memavailable_bytes()
        recoverable = recovered >= int(baseline * 0.85)
        ping_ok = bool(holder.ping().get("ok"))
    finally:
        holder.close()
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "kind": "h4_hold_calibration",
        "not_a_freeze": True,
        "capacity_id": cap,
        "boot_id": identity.get("boot_id"),
        "meminfo_kb": meminfo_kb(),
        "baseline_mem_available_bytes": baseline,
        "chunk_bytes": CHUNK_BYTES,
        "safety_floor_bytes": SAFETY_FLOOR_BYTES,
        "min_identifiable_drop_bytes": MIN_IDENTIFIABLE_DROP_BYTES,
        "high_bytes": chosen,
        "low_bytes": 0,
        "identifiable": bool(chosen > 0 and any(
            int(s.get("drop_from_baseline_bytes") or 0) >= MIN_IDENTIFIABLE_DROP_BYTES for s in steps
        )),
        "recovered_after_release": recoverable,
        "ping_ok_before_close": locals().get("ping_ok"),
        "recovered_mem_available_bytes": locals().get("recovered"),
        "steps": steps,
        "note": (
            "Dedicated CPU-page holder process. Co-running H4 pressure, not a static tight freeze. "
            "Do not enlarge the model. Do not treat this hold as G3."
        ),
    }
    dest = assert_write_path(repo_root() / "reports" / STUDY_ID / "g2" / "h4_calibrate.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload
