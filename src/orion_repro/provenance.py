"""Code / environment / manifest snapshots for run provenance (PLAN NFR01)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_source_tree(root: Path) -> dict[str, Any]:
    """Hash tracked implementation files. Does not require a git commit."""
    patterns = [
        root / "src" / "orion_repro",
        root / "tests",
        root / "configs",
        root / "experiments",
    ]
    extra = [
        root / "pyproject.toml",
        root / "requirements.lock.txt",
        root / "environment.lock.yml",
        root / "PLAN.md",
        root / "AGENTS.md",
    ]
    files: list[Path] = []
    for base in patterns:
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".json", ".md"})
        elif base.is_file():
            files.append(base)
    files.extend(p for p in extra if p.is_file())
    rels = sorted({p.resolve() for p in files}, key=lambda p: str(p.relative_to(root)))
    listing = []
    h = hashlib.sha256()
    for path in rels:
        digest = sha256_file(path)
        rel = str(path.relative_to(root)).replace("\\", "/")
        listing.append({"path": rel, "sha256": digest})
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("utf-8"))
        h.update(b"\n")
    return {
        "kind": "source_tree_sha256",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_sha256": h.hexdigest(),
        "n_files": len(listing),
        "files": listing,
    }


def file_sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


def fill_provenance(spec: dict[str, Any], *, root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Write snapshot ids into the resolved spec. Does not invent missing data manifests."""
    spec = dict(spec)
    spec["code_revision_or_snapshot"] = snapshot["aggregate_sha256"]
    lock = root / "requirements.lock.txt"
    spec["environment_lock_sha256"] = file_sha256_or_none(lock)
    manifest = spec.get("dataset", {}).get("split_manifest")
    if manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = root / path
        spec["dataset_manifest_sha256"] = file_sha256_or_none(path)
    elif spec.get("dataset_manifest_sha256") in (None, "null"):
        # Filled later from the frozen class-order dump if present.
        spec.setdefault("dataset_manifest_sha256", None)
    return spec


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
