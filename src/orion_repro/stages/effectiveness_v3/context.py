"""Directory, lock, and output isolation for effectiveness_v3."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion_repro.stages.effectiveness_v3.constants import (
    ALLOWED_WRITE_PREFIXES,
    DEFAULT_REVISION,
    HISTORICAL_GIT_PATHS,
    PROTECTED_PREFIXES,
    STUDY_ID,
    orion_python,
)
from orion_repro.stages.effectiveness_v3.util import StageError, sha256_file


@dataclass(frozen=True)
class StageContext:
    root: Path
    study_id: str = STUDY_ID
    revision: str = DEFAULT_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if self.study_id != STUDY_ID:
            raise StageError(f"StageContext study_id must be {STUDY_ID}")
        revision = str(self.revision).strip()
        if not revision or "/" in revision or ".." in revision:
            raise StageError(f"illegal revision {self.revision!r}")
        object.__setattr__(self, "revision", revision)

    @property
    def design_path(self) -> Path:
        return self.root / "experiments" / STUDY_ID / "design.yaml"

    @property
    def revision_dir(self) -> Path:
        return self.root / "experiments" / STUDY_ID / "revisions" / self.revision

    @property
    def config_dir(self) -> Path:
        return self.root / "configs" / STUDY_ID / self.revision

    @property
    def progress_path(self) -> Path:
        return self.root / "runs" / STUDY_ID / self.revision / "progress.json"

    @property
    def executor_dir(self) -> Path:
        return self.root / "runs" / STUDY_ID / self.revision / "executor"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports" / STUDY_ID / self.revision

    @property
    def project_lock(self) -> Path:
        return self.root / "runs" / ".orion_project.lock"

    def ensure_revision_dirs(self) -> None:
        for path in (
            self.revision_dir,
            self.config_dir / "dev",
            self.config_dir / "formal",
            self.progress_path.parent,
            self.executor_dir,
            self.report_dir,
        ):
            self.assert_output_path(path)
            path.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        resolved = Path(path).resolve()
        return resolved.relative_to(self.root).as_posix()

    def assert_output_path(self, path: Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        try:
            rel = resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise StageError(f"refusing path outside workspace: {resolved}") from exc
        if rel.startswith("../"):
            raise StageError(f"path traversal rejected: {rel}")
        for prefix in PROTECTED_PREFIXES:
            if rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
                raise StageError(f"refusing to write historical path {rel}")
        allowed = False
        if rel.startswith("runs/effectiveness_v3"):
            allowed = True
        for prefix in ALLOWED_WRITE_PREFIXES:
            if rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
                allowed = True
        if not allowed:
            raise StageError(f"output path is not in the effectiveness_v3 allowlist: {rel}")
        return resolved

    def reject_foreign_study(self, payload: dict[str, Any], *, field: str = "study_id") -> None:
        study = payload.get(field)
        if study != STUDY_ID:
            raise StageError(f"{field} {study!r} is not {STUDY_ID}; refusing old-stage objects")


def require_orion_interpreter(executable: str | None = None) -> Path:
    exe = Path(executable or sys.executable).resolve()
    if "mineru" in exe.parts or "envs/mineru" in exe.as_posix():
        raise StageError(f"interpreter is mineru: {exe}")
    if "orion" not in exe.as_posix():
        raise StageError(f"interpreter is not the orion env: {exe}")
    if exe.as_posix().startswith("/home/admin/"):
        raise StageError(f"interpreter is the archived WSL path: {exe}")
    return exe


def historical_protection_report(root: Path) -> dict[str, Any]:
    """Hash git-tracked historical files. Inspect-only; does not write."""
    root = Path(root).resolve()
    files: list[str] = []
    try:
        listed = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--", *HISTORICAL_GIT_PATHS],
            text=True,
        )
        files = [line.strip() for line in listed.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reason": f"git ls-files failed: {exc}", "n_files": 0, "dirty": []}
    dirty: list[str] = []
    try:
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--", *HISTORICAL_GIT_PATHS],
            text=True,
        )
        dirty = [line[3:].strip() for line in status.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        dirty = []
    digest = hashlib.sha256()
    n_hashed = 0
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\n")
        n_hashed += 1
    return {
        "ok": not dirty,
        "n_files": n_hashed,
        "aggregate_sha256": digest.hexdigest(),
        "dirty": dirty,
    }


def inspect_payload(context: StageContext) -> dict[str, Any]:
    from orion_repro.runner.reuse import platform_fingerprint
    from orion_repro.stages.effectiveness_v3.schema import load_design

    design = load_design(context.design_path)
    protection = historical_protection_report(context.root)
    return {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "executable": False,
        "design_version": design.get("design_version"),
        "formal_slots": design.get("formal_slots"),
        "design_path": str(context.design_path.relative_to(context.root)),
        "interpreter": str(Path(sys.executable).resolve()),
        "orion_python": str(orion_python()),
        "platform": platform_fingerprint(),
        "historical_protection": protection,
        "writes": False,
    }
