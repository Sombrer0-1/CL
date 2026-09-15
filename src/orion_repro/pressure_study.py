"""Independent pressure_v2 executor. Does not write light24 progress or reports."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orion_repro.runner.locks import acquire_locks
from orion_repro.runner.reuse import find_completed, reuse_identity
from orion_repro.runner.spec import load_yaml, validate_mapping

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "pressure_v2"
PROGRESS = ROOT / "runs" / "pressure_v2_progress.json"
EXECUTOR_DIR = ROOT / "runs" / "_pressure_v2_executor"
FROZEN_PATH = ROOT / "experiments" / "pressure_v2" / "frozen_protocol.json"


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)


def reconcile_active(state):
    active = state.get("active")
    if not active:
        return
    pid = active.get("pid")
    if pid and Path(f"/proc/{pid}").exists():
        raise RuntimeError(f"Unresolved prior child PID {pid}; inspect before resuming")
    final = ROOT / "runs" / active["run_id"] / "summary.json"
    outcome = json.loads(final.read_text()).get("status", "interrupted") if final.exists() else "interrupted"
    state["results"].append({**active, "status": outcome, "elapsed_s": None, "accounting": "unknown"})
    if not final.exists():
        append_outcome(active["run_id"], "interrupted", "Executor lost; elapsed time unknown")
    state["active"] = None


def stop_group(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def append_outcome(run_id, status, reason):
    with (ROOT / "experiments/registry.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "reason": reason,
                    "study_id": STUDY_ID,
                    "reconciled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )


def execute(path, spec, identity, state, state_path):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pressure_v2_" + uuid.uuid4().hex[:8]
    EXECUTOR_DIR.mkdir(parents=True, exist_ok=True)
    cfg = EXECUTOR_DIR / f"{run_id}.json"
    payload = {**spec, "run_id": run_id}
    cfg.write_text(json.dumps(payload))
    active = dict(config=str(Path(path).relative_to(ROOT) if Path(path).is_absolute() else path),
                  identity=identity, run_id=run_id, pid=None)
    state["active"] = active
    save(state_path, state)
    start = time.monotonic()
    status = "implementation_error"
    proc = None
    interrupted = False
    try:
        with (EXECUTOR_DIR / f"{run_id}.log").open("w") as log:
            env = dict(os.environ, WANDB_MODE="disabled", WANDB_DISABLED="true")
            if spec.get("training", {}).get("deterministic_algorithms"):
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            proc = subprocess.Popen(
                [sys.executable, "-m", "orion_repro.run", "--config", str(cfg)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            active["pid"] = proc.pid
            save(state_path, state)
            code = proc.wait()
            summary_path = ROOT / "runs" / run_id / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            status = summary.get("status", "implementation_error")
            if code != 0 and status == "completed":
                status = "implementation_error"
    except KeyboardInterrupt:
        interrupted = True
        if proc:
            stop_group(proc)
        status = "interrupted"
        append_outcome(run_id, status, "Study interrupted by signal")
    finally:
        if proc:
            stop_group(proc)
        elapsed = time.monotonic() - start
        state["consumed_s"] += elapsed
        row = {**active, "status": status, "elapsed_s": elapsed}
        state["results"].append(row)
        state["active"] = None
        save(state_path, state)
    print(json.dumps(row), flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return status


def skip_reason(spec, frozen):
    coverage = (frozen or {}).get("scenario_coverage") or {}
    group = spec.get("pressure_group")
    mapping = {
        "dyn": "dynamic_reservation",
        "decay": "dynamic_reservation",
        "pref": "preferences",
        "io": "io_prefetch",
    }
    key = mapping.get(group)
    if key and coverage.get(key) == "failed_to_construct":
        return "scenario_failed_to_construct"
    return None


def run_config_list(paths, specs, state, state_path, frozen=None):
    for path, spec in zip(paths, specs):
        skip = skip_reason(spec, frozen)
        if skip:
            print(json.dumps({"config": str(path), "status": skip}), flush=True)
            state["results"].append(
                {"config": str(path), "status": skip, "identity": None, "run_id": None, "elapsed_s": 0}
            )
            save(state_path, state)
            continue
        identity = reuse_identity(spec, ROOT)
        previous = find_completed(ROOT, identity)
        if previous:
            print(json.dumps({"config": str(path), "status": "reused", "run_id": previous["run_id"]}), flush=True)
            continue
        terminal = next(
            (
                item
                for item in reversed(state["results"])
                if item.get("identity") == identity
                and item.get("status")
                in {"cuda_oom", "host_oom", "budget_exceeded", "preflight_infeasible"}
            ),
            None,
        )
        if terminal:
            print(
                json.dumps(
                    {
                        "config": str(path),
                        "status": "previous_failure_preserved",
                        "outcome": terminal["status"],
                    }
                ),
                flush=True,
            )
            continue
        print(json.dumps({"starting": str(path), "consumed_s": state["consumed_s"]}), flush=True)
        status = execute(path, spec, identity, state, state_path)
        if status == "implementation_error":
            print(json.dumps({"paused": True, "reason": status}), flush=True)
            break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="experiments/pressure_v2/formal.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args(argv)
    matrix_path = ROOT / args.matrix if not Path(args.matrix).is_absolute() else Path(args.matrix)
    matrix = load_yaml(matrix_path)
    if matrix.get("study_id") != STUDY_ID:
        parser.error("Only pressure_v2 matrices are accepted")
    frozen = json.loads(FROZEN_PATH.read_text()) if FROZEN_PATH.exists() else None
    if matrix.get("name") == "formal_54":
        if not frozen:
            parser.error("formal matrix requires experiments/pressure_v2/frozen_protocol.json")
        if matrix.get("frozen_hash") != frozen.get("frozen_hash"):
            parser.error("matrix frozen_hash does not match frozen_protocol.json")
    paths = [ROOT / p for p in matrix["configs"]]
    specs = [load_yaml(p) for p in paths]
    for spec in specs:
        validate_mapping(spec, require_provenance=False)
        if spec.get("study_id") != STUDY_ID or spec.get("reuse_version") != 3:
            parser.error("Study configs must opt into pressure_v2 and reuse v3")
        if not reuse_identity(spec, ROOT):
            parser.error("Frozen dataset manifest required")
        if frozen and spec.get("phase") == "formal" and spec.get("frozen_hash") != frozen.get("frozen_hash"):
            parser.error("config frozen_hash mismatch")
    if args.dry_run:
        print(json.dumps({"matrix": args.matrix, "n_configs": len(paths), "frozen": bool(frozen)}, indent=2))
        return
    try:
        locks = acquire_locks(
            [ROOT / "runs" / ".orion_project.lock", ROOT / "runs" / ".light24.lock"]
        )
    except BlockingIOError:
        parser.error("Another Orion training executor is active")
    try:
        state = (
            json.loads(PROGRESS.read_text())
            if PROGRESS.exists()
            else {"study_id": STUDY_ID, "consumed_s": 0, "active": None, "results": []}
        )
        if state.get("study_id") != STUDY_ID:
            parser.error("progress file is not pressure_v2")
        reconcile_active(state)
        save(PROGRESS, state)

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        run_config_list(paths, specs, state, PROGRESS, frozen=frozen)
        print(json.dumps({"consumed_s": state["consumed_s"], "n_results": len(state["results"])}))
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    main()
