"""Fresh-process executor for effectiveness_v3. Does not write historical progress files."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion_repro.runner.locks import acquire_locks
from orion_repro.runner.reuse import find_completed, reuse_identity
from orion_repro.runner.spec import load_yaml, validate_mapping
from orion_repro.stages.effectiveness_v3.constants import STUDY_ID, orion_python
from orion_repro.stages.effectiveness_v3.context import StageContext, require_orion_interpreter
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json


TERMINAL_RESOURCE = {"cuda_oom", "host_oom", "budget_exceeded", "preflight_infeasible"}


def process_signature(pid: int) -> dict[str, str] | None:
    stat_path = Path(f"/proc/{pid}/stat")
    cmd_path = Path(f"/proc/{pid}/cmdline")
    if not stat_path.exists():
        return None
    try:
        raw = stat_path.read_text(encoding="utf-8")
        comm_end = raw.rfind(")")
        fields = raw[comm_end + 2 :].split()
        starttime = fields[19]
        cmdline = cmd_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except (OSError, IndexError):
        return None
    return {"pid": str(pid), "starttime": starttime, "cmdline": cmdline}


def signatures_match(saved: dict[str, Any] | None, live: dict[str, str] | None) -> bool:
    if not saved or not live:
        return False
    return (
        str(saved.get("pid")) == str(live.get("pid"))
        and str(saved.get("starttime")) == str(live.get("starttime"))
        and str(saved.get("cmdline") or "") == str(live.get("cmdline") or "")
    )


def save_progress(path: Path, state: dict[str, Any], context: StageContext) -> None:
    context.assert_output_path(path)
    atomic_write_json(path, state)


def reconcile_active(state: dict[str, Any], context: StageContext) -> None:
    active = state.get("active")
    if not active:
        return
    sig = process_signature(int(active["pid"])) if active.get("pid") else None
    if signatures_match(active.get("process"), sig):
        raise RuntimeError(f"Unresolved prior child PID {active['pid']}; inspect before resuming")
    run_id = active.get("run_id")
    final = context.root / "runs" / run_id / "summary.json" if run_id else None
    outcome = "interrupted"
    if final and final.exists():
        outcome = json.loads(final.read_text(encoding="utf-8")).get("status", "interrupted")
    state["results"].append({**active, "status": outcome, "elapsed_s": None, "accounting": "unknown"})
    if not final or not final.exists():
        _append_registry(context, {"run_id": run_id, "status": "interrupted", "reason": "Executor lost; elapsed time unknown"})
    state["active"] = None


def stop_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _append_registry(context: StageContext, row: dict[str, Any]) -> None:
    path = context.root / "experiments" / "registry.jsonl"
    context.assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**row, "study_id": STUDY_ID, "revision": context.revision}, default=str) + "\n")


def execute(path: Path, spec: dict[str, Any], identity: str | None, state: dict[str, Any], context: StageContext) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"effectiveness_v3_{stamp}_{uuid.uuid4().hex[:8]}"
    context.ensure_revision_dirs()
    cfg = context.executor_dir / f"{run_id}.json"
    context.assert_output_path(cfg)
    payload = {**spec, "run_id": run_id}
    cfg.write_text(json.dumps(payload, default=str), encoding="utf-8")
    python = require_orion_interpreter(str(orion_python()))
    active = {
        "config": str(path.relative_to(context.root) if path.is_absolute() else path),
        "identity": identity,
        "run_id": run_id,
        "pid": None,
        "process": None,
        "cell_id": spec.get("v3_cell_id"),
    }
    state["active"] = active
    save_progress(context.progress_path, state, context)
    start = time.monotonic()
    status = "implementation_error"
    proc = None
    interrupted = False
    try:
        log_path = context.executor_dir / f"{run_id}.log"
        context.assert_output_path(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            env = dict(os.environ, WANDB_MODE="disabled", WANDB_DISABLED="true")
            if spec.get("training", {}).get("deterministic_algorithms"):
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            proc = subprocess.Popen(
                [str(python), "-m", "orion_repro.run", "--config", str(cfg)],
                cwd=context.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            active["pid"] = proc.pid
            active["process"] = process_signature(proc.pid)
            save_progress(context.progress_path, state, context)
            code = proc.wait()
            summary_path = context.root / "runs" / run_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            status = summary.get("status", "implementation_error")
            if code != 0 and status == "completed":
                status = "implementation_error"
    except KeyboardInterrupt:
        interrupted = True
        if proc:
            stop_group(proc)
        status = "interrupted"
        _append_registry(context, {"run_id": run_id, "status": status, "reason": "Study interrupted by signal"})
    finally:
        if proc:
            stop_group(proc)
        elapsed = time.monotonic() - start
        state["consumed_s"] += elapsed
        row = {**active, "status": status, "elapsed_s": elapsed}
        state["results"].append(row)
        state["active"] = None
        save_progress(context.progress_path, state, context)
    print(json.dumps(row, default=str), flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return status


def skip_reason(spec: dict[str, Any], frozen: dict[str, Any] | None) -> str | None:
    coverage = (frozen or {}).get("scenario_coverage") or {}
    group = spec.get("v3_group")
    mapping = {
        "C": "dynamic_reservation",
        "D": "S06",
        "E": "io_prefetch",
        "H": "host",
    }
    key = mapping.get(group)
    if key and coverage.get(key) in {"failed_to_construct", "unavailable"}:
        return "scenario_not_realized"
    if group == "H" and (frozen or {}).get("host", {}).get("status") not in {"available", "realized"}:
        return "scenario_not_realized"
    if group == "E" and not (frozen or {}).get("io_prefetch", {}).get("constructed"):
        return "scenario_not_realized"
    return None


def run_config_list(
    paths: list[Path],
    specs: list[dict[str, Any]],
    state: dict[str, Any],
    context: StageContext,
    frozen: dict[str, Any] | None = None,
) -> None:
    for path, spec in zip(paths, specs):
        skip = skip_reason(spec, frozen)
        if skip:
            print(json.dumps({"config": str(path), "status": skip}), flush=True)
            state["results"].append(
                {"config": str(path), "status": skip, "identity": None, "run_id": None, "elapsed_s": 0, "cell_id": spec.get("v3_cell_id")}
            )
            save_progress(context.progress_path, state, context)
            continue
        identity = reuse_identity(spec, context.root)
        previous = find_completed(context.root, identity)
        rel = str(path.relative_to(context.root) if path.is_absolute() else path)
        if previous:
            row = {
                "config": rel,
                "status": "reused",
                "run_id": previous["run_id"],
                "identity": identity,
                "elapsed_s": 0,
                "cell_id": spec.get("v3_cell_id"),
            }
            print(json.dumps(row), flush=True)
            state["results"].append(row)
            save_progress(context.progress_path, state, context)
            continue
        terminal = next(
            (
                item
                for item in reversed(state["results"])
                if item.get("identity") == identity and item.get("status") in TERMINAL_RESOURCE
            ),
            None,
        )
        if terminal:
            row = {
                "config": rel,
                "status": "previous_failure_preserved",
                "outcome": terminal["status"],
                "run_id": terminal.get("run_id"),
                "identity": identity,
                "elapsed_s": 0,
                "cell_id": spec.get("v3_cell_id"),
            }
            print(json.dumps(row), flush=True)
            state["results"].append(row)
            save_progress(context.progress_path, state, context)
            continue
        print(json.dumps({"starting": str(path), "consumed_s": state["consumed_s"]}), flush=True)
        status = execute(path, spec, identity, state, context)
        if status == "implementation_error":
            print(json.dumps({"paused": True, "reason": status}), flush=True)
            break


def dry_run_matrix(matrix: dict[str, Any], context: StageContext) -> dict[str, Any]:
    context.reject_foreign_study(matrix)
    if matrix.get("kind") not in {"matrix_manifest", None}:
        if matrix.get("study_id") != STUDY_ID:
            raise StageError("foreign matrix")
    return {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "n_configs": len(matrix.get("configs") or []),
        "eligible_n": matrix.get("eligible_n"),
        "excluded_n": matrix.get("excluded_n"),
        "design_slots": matrix.get("design_slots"),
        "writes": False,
    }


def execute_matrix(matrix: dict[str, Any], context: StageContext, *, frozen: dict[str, Any] | None) -> dict[str, Any]:
    context.reject_foreign_study(matrix)
    require_orion_interpreter()
    if matrix.get("frozen_hash") and frozen and matrix["frozen_hash"] != frozen.get("frozen_hash"):
        raise StageError("matrix frozen_hash does not match frozen protocol")
    paths = [context.root / p for p in matrix.get("configs") or []]
    specs = [load_yaml(p) for p in paths]
    for spec in specs:
        validate_mapping(spec, require_provenance=False)
        if spec.get("study_id") != STUDY_ID or spec.get("reuse_version") != 4:
            raise StageError("Study configs must opt into effectiveness_v3 and reuse v4")
        if frozen and spec.get("phase") == "formal" and spec.get("frozen_hash") != frozen.get("frozen_hash"):
            raise StageError("config frozen_hash mismatch")
    try:
        locks = acquire_locks([context.project_lock])
    except BlockingIOError as exc:
        raise StageError("Another Orion training executor is active") from exc
    try:
        state = (
            json.loads(context.progress_path.read_text(encoding="utf-8"))
            if context.progress_path.exists()
            else {"study_id": STUDY_ID, "revision": context.revision, "consumed_s": 0, "active": None, "results": []}
        )
        if state.get("study_id") != STUDY_ID:
            raise StageError("progress file is not effectiveness_v3")
        reconcile_active(state, context)
        save_progress(context.progress_path, state, context)

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        run_config_list(paths, specs, state, context, frozen=frozen)
        return {"consumed_s": state["consumed_s"], "n_results": len(state["results"])}
    finally:
        for handle in locks:
            handle.close()
