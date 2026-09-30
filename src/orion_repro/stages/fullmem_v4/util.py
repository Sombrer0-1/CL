"""Atomic JSON and path guards for fullmem_v4."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import ALLOWED_WRITE_PREFIXES, PROTECTED_PREFIXES, STUDY_ID, repo_root


class StageError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_to_root(path: Path, root: Path | None = None) -> str:
    root = (root or repo_root()).resolve()
    return str(Path(path).resolve().relative_to(root)).replace("\\", "/")


def assert_write_path(path: Path, root: Path | None = None) -> Path:
    root = (root or repo_root()).resolve()
    resolved = Path(path).resolve()
    try:
        rel = rel_to_root(resolved, root)
    except ValueError as exc:
        raise StageError(f"refusing write outside repo: {resolved}") from exc
    if any(rel == p.rstrip("/") or rel.startswith(p) for p in PROTECTED_PREFIXES):
        raise StageError(f"refusing write into protected historical path: {rel}")
    if not any(rel == p.rstrip("/") or rel.startswith(p) for p in ALLOWED_WRITE_PREFIXES):
        raise StageError(f"fullmem_v4 may only write under v4 prefixes, got {rel}")
    return resolved


def require_orion_python(executable: str | None = None) -> str:
    from orion_repro.stages.fullmem_v4.constants import ORION_PYTHON

    path = executable or ORION_PYTHON
    if Path(path).resolve() != Path(ORION_PYTHON).resolve():
        raise StageError(f"refusing non-orion interpreter {path}; required {ORION_PYTHON}")
    if not Path(path).is_file():
        raise StageError(f"orion interpreter missing: {path}")
    return path


def require_study(payload: dict[str, Any]) -> None:
    if payload.get("study_id") != STUDY_ID:
        raise StageError(f"payload study_id must be {STUDY_ID}, got {payload.get('study_id')!r}")
