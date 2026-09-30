"""Persistent task queue. Duplicate task_id with the same hash is idempotent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import ORION_PYTHON, STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.util import (
    StageError,
    append_jsonl,
    assert_write_path,
    atomic_write_json,
    content_hash,
    load_json,
    require_study,
    sha256_file,
    utc_now,
)

TERMINAL = {
    "completed",
    "resource_failure",
    "implementation_error",
    "infrastructure_failure",
    "safety_stopped",
    "unknown",
    "rejected",
}


def stage_paths(root: Path | None = None) -> dict[str, Path]:
    root = (root or repo_root()).resolve()
    base = root / "experiments" / STUDY_ID
    paths = {
        "root": root,
        "base": base,
        "queue": base / "queue",
        "inbox": base / "queue" / "inbox",
        "running": base / "queue" / "running",
        "done": base / "queue" / "done",
        "rejected": base / "queue" / "rejected",
        "state": base / "state" / "executor.json",
        "events": base / "state" / "events.jsonl",
        "lock": root / "runs" / ".orion_project.lock",
        "runs": root / "runs" / STUDY_ID,
        "reports": root / "reports" / STUDY_ID,
    }
    return paths


def ensure_dirs(root: Path | None = None) -> dict[str, Path]:
    paths = stage_paths(root)
    for key in ("inbox", "running", "done", "rejected", "runs", "reports"):
        assert_write_path(paths[key], paths["root"]).mkdir(parents=True, exist_ok=True)
    assert_write_path(paths["state"].parent, paths["root"]).mkdir(parents=True, exist_ok=True)
    return paths


def _config_content_hashes(argv: list[str], root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for index, token in enumerate(argv):
        if token != "--config" or index + 1 >= len(argv):
            continue
        rel = argv[index + 1].replace("\\", "/")
        path = Path(rel)
        if not path.is_absolute():
            path = root / rel
        if not path.is_file():
            raise StageError(f"config file missing for task identity: {rel}")
        hashes[rel] = sha256_file(path)
    return hashes


def _task_body_for_hash(task: dict[str, Any]) -> dict[str, Any]:
    skip = {"content_hash", "submitted_at_utc", "status", "attempt"}
    return {k: v for k, v in task.items() if k not in skip}


def normalize_task(raw: dict[str, Any]) -> dict[str, Any]:
    require_study(raw)
    task_id = str(raw.get("task_id") or "").strip()
    if not task_id or "/" in task_id or ".." in task_id:
        raise StageError(f"illegal task_id {task_id!r}")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise StageError("argv must be a non-empty list of strings")
    interpreter = str(raw.get("interpreter") or ORION_PYTHON)
    if interpreter != ORION_PYTHON:
        raise StageError(f"interpreter must be {ORION_PYTHON}")
    task = {
        "study_id": STUDY_ID,
        "design_version": raw.get("design_version") or "design-v1",
        "task_id": task_id,
        "seq": int(raw.get("seq") or 0),
        "kind": str(raw.get("kind") or "probe"),
        "argv": list(argv),
        "interpreter": interpreter,
        "working_directory": str(raw.get("working_directory") or str(repo_root())),
        "required_capacity": str(raw.get("required_capacity") or "any"),
        "batch_id": str(raw.get("batch_id") or "default"),
        "timeout_s": raw.get("timeout_s"),
        "env": dict(raw.get("env") or {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}),
    }
    if task["kind"] not in {"probe", "train", "dummy"}:
        raise StageError(f"unsupported kind {task['kind']}")
    _assert_train_admitted(task)
    task["config_content_hashes"] = _config_content_hashes(task["argv"], Path(task["working_directory"]))
    task["content_hash"] = content_hash(_task_body_for_hash(task))
    return task


def _assert_train_admitted(task: dict[str, Any]) -> None:
    """kind=train is admitted only for batches named by the G3 freeze file."""
    if task["kind"] != "train":
        return
    path = repo_root() / "reports" / STUDY_ID / "g3" / "FREEZE.json"
    if not path.is_file():
        raise StageError("formal train tasks are refused until G3 freeze")
    freeze = load_json(path)
    if freeze.get("admits_kind_train") is not True:
        raise StageError("formal train tasks are refused until G3 freeze")
    admitted = {str(item) for item in (freeze.get("admitted_batches") or [])}
    if task["batch_id"] not in admitted:
        raise StageError(f"batch {task['batch_id']} is outside the G3 freeze")
    cap = str(freeze.get("capacity_id") or "")
    if task["required_capacity"] != cap:
        raise StageError(f"train capacity {task['required_capacity']} != freeze {cap}")


def _existing_record(paths: dict[str, Path], task_id: str) -> tuple[Path, dict[str, Any]] | None:
    for bucket in ("inbox", "running", "done", "rejected"):
        path = paths[bucket] / f"{task_id}.json"
        if path.exists():
            return path, load_json(path)
    return None


def submit(raw: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    paths = ensure_dirs(root)
    task = normalize_task(raw)
    found = _existing_record(paths, task["task_id"])
    if found:
        path, existing = found
        existing_hash = existing.get("content_hash") or content_hash(_task_body_for_hash(existing))
        if existing_hash == task["content_hash"]:
            return {
                "accepted": True,
                "idempotent": True,
                "task_id": task["task_id"],
                "path": str(path),
                "status": existing.get("status") or "admitted",
            }
        raise StageError(f"task_id {task['task_id']} exists with different content_hash")
    record = {
        **task,
        "status": "admitted",
        "submitted_at_utc": utc_now(),
        "attempt": 1,
    }
    dest = assert_write_path(paths["inbox"] / f"{task['task_id']}.json", paths["root"])
    atomic_write_json(dest, record)
    append_jsonl(
        assert_write_path(paths["events"], paths["root"]),
        {"ts": utc_now(), "event": "submitted", "task_id": task["task_id"], "content_hash": task["content_hash"]},
    )
    return {"accepted": True, "idempotent": False, "task_id": task["task_id"], "path": str(dest), "status": "admitted"}


def list_inbox(root: Path | None = None) -> list[dict[str, Any]]:
    paths = ensure_dirs(root)
    tasks = []
    for path in sorted(paths["inbox"].glob("*.json")):
        task = load_json(path)
        task["_path"] = str(path)
        tasks.append(task)
    tasks.sort(key=lambda item: (int(item.get("seq") or 0), str(item.get("task_id"))))
    return tasks


def default_state() -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "mode": "idle",
        "pause_reason": None,
        "batch_id": None,
        "boot_id_at_batch_start": None,
        "current_task_id": None,
        "worker_unit": None,
        "worker_pid": None,
        "heartbeat_ts": None,
        "completed_n": 0,
        "failed_n": 0,
        "last_event": None,
    }


def load_state(root: Path | None = None) -> dict[str, Any]:
    paths = ensure_dirs(root)
    if not paths["state"].exists():
        return default_state()
    state = load_json(paths["state"])
    require_study(state)
    return state


def save_state(state: dict[str, Any], root: Path | None = None) -> None:
    paths = ensure_dirs(root)
    require_study(state)
    atomic_write_json(assert_write_path(paths["state"], paths["root"]), state)
