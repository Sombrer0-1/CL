"""Detached systemd-facing queue worker. Never auto-retries a failed attempt."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from orion_repro.runner.locks import acquire_locks
from orion_repro.stages.fullmem_v4.constants import (
    HEARTBEAT_S,
    IDLE_SLEEP_S,
    MIN_FREE_BYTES,
    ORION_PYTHON,
    TASK_UNIT_PREFIX,
    repo_root,
)
from orion_repro.stages.fullmem_v4.identity import capacity_matches, collect_identity
from orion_repro.stages.fullmem_v4.queue import ensure_dirs, list_inbox, load_state, save_state, stage_paths
from orion_repro.stages.fullmem_v4.util import StageError, append_jsonl, assert_write_path, atomic_write_json, load_json, utc_now
import shutil


def _event(paths: dict[str, Path], **payload: Any) -> None:
    append_jsonl(assert_write_path(paths["events"], paths["root"]), {"ts": utc_now(), **payload})


def _move_task(task: dict[str, Any], dest_dir: Path, *, status: str, extra: dict[str, Any] | None = None) -> Path:
    root = Path(task.get("_root") or repo_root())
    paths = stage_paths(root)
    src = Path(task["_path"])
    dest = assert_write_path(dest_dir / f"{task['task_id']}.json", paths["root"])
    record = {k: v for k, v in task.items() if not k.startswith("_")}
    record["status"] = status
    record["updated_at_utc"] = utc_now()
    if extra:
        record.update(extra)
    atomic_write_json(dest, record)
    if src.exists() and src.resolve() != dest.resolve():
        src.unlink()
    return dest


def pause(reason: str, root: Path | None = None, *, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = ensure_dirs(root)
    state = load_state(root)
    state["mode"] = "paused"
    state["pause_reason"] = reason
    state["heartbeat_ts"] = utc_now()
    if identity:
        state["last_identity"] = {
            "boot_id": identity.get("boot_id"),
            "capacity_id": (identity.get("capacity") or {}).get("capacity_id"),
        }
    save_state(state, root)
    _event(paths, event="paused", reason=reason)
    return state


def resume(root: Path | None = None) -> dict[str, Any]:
    state = load_state(root)
    if state.get("current_task_id"):
        raise StageError("cannot resume while a worker identity is still recorded; inspect the running task")
    state["mode"] = "idle"
    state["pause_reason"] = None
    state["heartbeat_ts"] = utc_now()
    save_state(state, root)
    _event(ensure_dirs(root), event="resumed")
    return state


def _disk_ok(root: Path) -> tuple[bool, int]:
    free = shutil.disk_usage(root).free
    return free >= MIN_FREE_BYTES, int(free)


HEALTH_SETTLE_S = 2.0


def _running_records(root: Path) -> list[Path]:
    return sorted(stage_paths(root)["running"].glob("*.json"))


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    return True


def _unit_is_active(unit: str | None) -> bool:
    if not unit:
        return False
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", str(unit)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (proc.stdout or "").strip() == "active"


def list_active_task_units() -> list[str]:
    try:
        proc = subprocess.run(
            [
                "systemctl",
                "--user",
                "list-units",
                "--type=service",
                "--state=active",
                "--plain",
                "--no-legend",
                f"{TASK_UNIT_PREFIX}*.service",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    units: list[str] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if name.startswith(TASK_UNIT_PREFIX) and name.endswith(".service"):
            units.append(name)
    return units


def health_check(root: Path, identity: dict[str, Any] | None = None) -> tuple[bool, str]:
    identity = identity or collect_identity(include_cuda=False)
    ok, free = _disk_ok(root)
    if not ok:
        return False, f"disk_low:{free}"
    if not identity.get("swap_is_zero", True):
        return False, "swap_nonzero"
    mem = identity.get("meminfo_kb") or {}
    if mem.get("MemAvailable") is None:
        return False, "memavailable_missing"
    units = list_active_task_units()
    if units:
        return False, "task_units_active:" + ",".join(units)
    if _running_records(root):
        return False, "running_records_present"
    return True, "ok"


def _wait_unit_inactive(unit: str, state: dict[str, Any], root: Path) -> None:
    last = time.monotonic()
    while _unit_is_active(unit):
        if time.monotonic() - last >= HEARTBEAT_S:
            state["heartbeat_ts"] = utc_now()
            save_state(state, root)
            last = time.monotonic()
        time.sleep(0.5)


def _load_running_task(root: Path, task_id: str | None) -> dict[str, Any] | None:
    paths = stage_paths(root)
    candidates = []
    if task_id:
        path = paths["running"] / f"{task_id}.json"
        if path.is_file():
            candidates.append(path)
    candidates.extend(_running_records(root))
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        task = load_json(path)
        task["_path"] = str(path)
        task["_root"] = str(root)
        return task
    return None


def reconcile_open_work(state: dict[str, Any], root: Path) -> dict[str, Any] | None:
    """Inspect recorded workers before taking a new inbox item.

    Returns None when the supervisor may proceed to inbox. Otherwise returns
    an action dict: wait for a live unit, finalize a dead leftover from marker/log,
    or pause if the leftover cannot be identified.
    """
    current = state.get("current_task_id")
    unit = state.get("worker_unit")
    pid = state.get("worker_pid")
    running = _running_records(root)
    live_unit = _unit_is_active(str(unit) if unit else None)
    live_pid = _pid_alive(pid)
    leftover_units = [name for name in list_active_task_units() if name != unit]
    if leftover_units and not live_unit:
        return {"action": "pause", "reason": "orphan_task_units:" + ",".join(leftover_units)}
    if live_unit or live_pid:
        task = _load_running_task(root, str(current) if current else None)
        if task is None:
            return {"action": "pause", "reason": f"live_worker_without_running_record:{unit or pid}"}
        return {"action": "wait", "task": task, "unit": unit or f"{TASK_UNIT_PREFIX}{task['task_id']}.service"}
    if current or running or state.get("mode") == "running":
        task = _load_running_task(root, str(current) if current else None)
        if task is None:
            return {"action": "pause", "reason": f"stale_running_task:{current or 'unknown'}"}
        return {
            "action": "finalize_dead",
            "task": task,
            "unit": unit or f"{TASK_UNIT_PREFIX}{task['task_id']}.service",
        }
    return None


def _marker_path(task: dict[str, Any], root: Path) -> Path | None:
    argv = list(task.get("argv") or [])
    if "--marker" not in argv:
        return None
    rel = argv[argv.index("--marker") + 1]
    path = Path(rel)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path(str(task.get("working_directory") or root)) / rel)
        candidates.append(root / rel)
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0] if candidates else None


def infer_exit_code(
    task: dict[str, Any],
    root: Path,
    log_text: str,
    *,
    code: int | None,
) -> int | None:
    if code is not None:
        return int(code)
    marker = _marker_path(task, root)
    if marker is not None and marker.is_file():
        return 0
    blob = log_text.lower()
    if '"ok": true' in blob:
        return 0
    return None


def _finalize_existing_task(task: dict[str, Any], state: dict[str, Any], identity: dict[str, Any], root: Path, *, code: int | None, log_text: str) -> str:
    paths = stage_paths(root)
    code = infer_exit_code(task, root, log_text, code=code)
    status = _classify_exit(code, log_text)
    extra = {"exit_code": code, "log": str(paths["runs"] / "executor" / f"{task['task_id']}.log"), "finished_at_utc": utc_now(), "adopted_after_supervisor_restart": True}
    _move_task(task, paths["done"], status=status, extra=extra)
    state["current_task_id"] = None
    state["worker_pid"] = None
    state["worker_unit"] = None
    if status == "completed":
        state["completed_n"] = int(state.get("completed_n") or 0) + 1
        state["mode"] = "idle"
        save_state(state, root)
        _event(paths, event="completed", task_id=task["task_id"], exit_code=code, adopted=True)
        return status
    state["failed_n"] = int(state.get("failed_n") or 0) + 1
    save_state(state, root)
    if status == "resource_failure":
        return _after_resource_failure(task, state, identity, root, code)
    pause(f"{status}:{task['task_id']}", root, identity=identity)
    _event(paths, event="paused_on_failure", task_id=task["task_id"], status=status, exit_code=code, adopted=True)
    return "paused"


def _after_resource_failure(task: dict[str, Any], state: dict[str, Any], identity: dict[str, Any], root: Path, code: int | None) -> str:
    paths = stage_paths(root)
    unit = state.get("worker_unit") or f"{TASK_UNIT_PREFIX}{task['task_id']}.service"
    _wait_unit_inactive(str(unit), state, root)
    time.sleep(HEALTH_SETTLE_S)
    fresh = collect_identity(include_cuda=False)
    ok, reason = health_check(root, fresh)
    if not ok:
        pause(f"health_after_resource_failure:{reason}", root, identity=fresh)
        _event(paths, event="paused_after_resource_failure", task_id=task["task_id"], health=reason, exit_code=code)
        return "paused"
    state["mode"] = "idle"
    save_state(state, root)
    _event(paths, event="resource_failure", task_id=task["task_id"], exit_code=code, health=reason)
    return "resource_failure"


def adopt_or_wait(state: dict[str, Any], identity: dict[str, Any], root: Path, decision: dict[str, Any]) -> str:
    task = decision["task"]
    unit = str(decision.get("unit") or f"{TASK_UNIT_PREFIX}{task['task_id']}.service")
    _event(stage_paths(root), event="adopted_worker", task_id=task["task_id"], unit=unit, kind=decision.get("action"))
    if decision.get("action") != "finalize_dead":
        _wait_unit_inactive(unit, state, root)
    log_path = stage_paths(root)["runs"] / "executor" / f"{task['task_id']}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return _finalize_existing_task(task, state, identity, root, code=None, log_text=log_text)


def _handle_open_work(state: dict[str, Any], identity: dict[str, Any], root: Path, decision: dict[str, Any]) -> str:
    keep_pause = state.get("mode") == "paused"
    pause_reason = state.get("pause_reason")
    status = adopt_or_wait(state, identity, root, decision)
    if keep_pause:
        pause(str(pause_reason or "supervisor_signal"), root, identity=identity)
        return "paused"
    return status


def _reconcile_boot(state: dict[str, Any], identity: dict[str, Any], root: Path) -> str | None:
    previous = state.get("boot_id_at_batch_start")
    current = identity.get("boot_id")
    if not (previous and current and previous != current):
        return None
    open_work = bool(
        state.get("current_task_id")
        or state.get("mode") == "running"
        or _running_records(root)
        or list_inbox(root)
    )
    # Advance boot_id in one shot. Otherwise host_reboot_with_open_work would
    # fire every loop and starve finalize_dead / evidence recovery.
    state["boot_id_at_batch_start"] = current
    if not open_work:
        state["batch_id"] = None
        state["mode"] = "idle"
        save_state(state, root)
        return None
    save_state(state, root)
    return "host_reboot_with_open_work"


def _systemd_cmd(task: dict[str, Any]) -> list[str]:
    unit = f"{TASK_UNIT_PREFIX}{task['task_id']}.service"
    argv = list(task["argv"])
    if argv[0] != ORION_PYTHON:
        argv = [ORION_PYTHON, *argv]
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={unit}",
        f"--working-directory={task['working_directory']}",
        "-p",
        "KillMode=mixed",
        "-p",
        "OOMPolicy=stop",
        "--setenv=WANDB_MODE=disabled",
        "--setenv=WANDB_DISABLED=true",
        "--setenv=PYTHONUNBUFFERED=1",
        f"--setenv=PYTHONPATH={repo_root() / 'src'}",
        "--",
        *argv,
    ]


def _classify_exit(code: int | None, log_text: str) -> str:
    blob = log_text.lower()
    if "h4 scene_failure" in blob or "scene_failure" in blob:
        return "resource_failure"
    if "oom-kill" in blob or "outofmemoryerror" in blob or "cuda out of memory" in blob or "cuda error: out of memory" in blob or "cuda_error_out_of_memory" in blob:
        return "resource_failure"
    if code == 0:
        return "completed"
    if code in {137, 9}:
        return "unknown"
    return "implementation_error"


def process_one(task: dict[str, Any], state: dict[str, Any], identity: dict[str, Any], root: Path) -> str:
    paths = stage_paths(root)
    required = str(task.get("required_capacity") or "any")
    if not capacity_matches(required, identity):
        pause(
            f"capacity_mismatch:{required}:{(identity.get('capacity') or {}).get('capacity_id')}",
            root,
            identity=identity,
        )
        return "paused"
    ok, free = _disk_ok(root)
    if not ok:
        pause(f"disk_low:{free}", root, identity=identity)
        return "paused"
    running_path = _move_task(task, paths["running"], status="running", extra={"boot_id": identity.get("boot_id")})
    task["_path"] = str(running_path)
    log_path = assert_write_path(paths["runs"] / "executor" / f"{task['task_id']}.log", root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not state.get("boot_id_at_batch_start"):
        state["boot_id_at_batch_start"] = identity.get("boot_id")
    state["mode"] = "running"
    state["current_task_id"] = task["task_id"]
    state["batch_id"] = task.get("batch_id")
    state["heartbeat_ts"] = utc_now()
    save_state(state, root)
    _event(paths, event="started", task_id=task["task_id"])
    cmd = _systemd_cmd(task)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=task["working_directory"], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        state["worker_pid"] = proc.pid
        state["worker_unit"] = f"{TASK_UNIT_PREFIX}{task['task_id']}.service"
        save_state(state, root)
        last = time.monotonic()
        while proc.poll() is None:
            if time.monotonic() - last >= HEARTBEAT_S:
                state["heartbeat_ts"] = utc_now()
                save_state(state, root)
                last = time.monotonic()
            time.sleep(0.5)
        code = proc.returncode
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    status = _classify_exit(code, log_text)
    extra = {"exit_code": code, "log": str(log_path), "finished_at_utc": utc_now()}
    _move_task(task, paths["done"], status=status, extra=extra)
    state["current_task_id"] = None
    state["worker_pid"] = None
    state["worker_unit"] = None
    if status == "completed":
        state["completed_n"] = int(state.get("completed_n") or 0) + 1
        state["mode"] = "idle"
        save_state(state, root)
        _event(paths, event="completed", task_id=task["task_id"], exit_code=code)
        return status
    state["failed_n"] = int(state.get("failed_n") or 0) + 1
    save_state(state, root)
    if status == "resource_failure":
        return _after_resource_failure(task, state, identity, root, code)
    pause(f"{status}:{task['task_id']}", root, identity=identity)
    _event(paths, event="paused_on_failure", task_id=task["task_id"], status=status, exit_code=code)
    return "paused"


def executor_once(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    paths = ensure_dirs(root)
    state = load_state(root)
    identity = collect_identity(include_cuda=False)
    reboot_reason = _reconcile_boot(state, identity, root)
    if reboot_reason:
        pause(reboot_reason, root, identity=identity)
        return {"action": "paused", "reason": reboot_reason}
    if state.get("mode") == "paused":
        open_work = reconcile_open_work(state, root)
        if open_work and open_work.get("action") in {"wait", "finalize_dead"}:
            try:
                locks = acquire_locks([paths["lock"]])
            except BlockingIOError:
                state["heartbeat_ts"] = utc_now()
                save_state(state, root)
                return {"action": "paused", "reason": state.get("pause_reason")}
            try:
                status = _handle_open_work(state, identity, root, open_work)
                return {"action": status, "task_id": open_work["task"]["task_id"], "adopted": True, "kept_pause": True}
            finally:
                for handle in locks:
                    handle.close()
        state["heartbeat_ts"] = utc_now()
        save_state(state, root)
        return {"action": "paused", "reason": state.get("pause_reason")}
    open_work = reconcile_open_work(state, root)
    if open_work:
        if open_work.get("action") == "pause":
            pause(str(open_work.get("reason") or "open_work"), root, identity=identity)
            return {"action": "paused", "reason": open_work.get("reason")}
        if open_work.get("action") in {"wait", "finalize_dead"}:
            try:
                locks = acquire_locks([paths["lock"]])
            except BlockingIOError:
                pause("project_lock_busy", root, identity=identity)
                return {"action": "paused", "reason": "project_lock_busy"}
            try:
                status = _handle_open_work(state, identity, root, open_work)
                return {"action": status, "task_id": open_work["task"]["task_id"], "adopted": True}
            finally:
                for handle in locks:
                    handle.close()
    tasks = list_inbox(root)
    if not tasks:
        state["mode"] = "idle"
        state["heartbeat_ts"] = utc_now()
        save_state(state, root)
        return {"action": "idle"}
    task = tasks[0]
    task["_root"] = str(root)
    try:
        locks = acquire_locks([paths["lock"]])
    except BlockingIOError:
        pause("project_lock_busy", root, identity=identity)
        return {"action": "paused", "reason": "project_lock_busy"}
    try:
        status = process_one(task, state, identity, root)
        return {"action": status, "task_id": task["task_id"]}
    finally:
        for handle in locks:
            handle.close()


def executor_loop(*, once: bool = False, root: Path | None = None) -> int:
    root = (root or repo_root()).resolve()

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    while True:
        try:
            result = executor_once(root)
        except KeyboardInterrupt:
            pause("supervisor_signal", root)
            return 0
        if once:
            return 0
        time.sleep(IDLE_SLEEP_S if result.get("action") in {"idle", "paused"} else 0.2)
