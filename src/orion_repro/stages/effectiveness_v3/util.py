"""Shared hashing and atomic JSON helpers for effectiveness_v3."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


class StageError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StageError(f"{path} did not contain a mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def reject_nonfinite(obj: Any, prefix: str = "value") -> None:
    if obj is None:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        raise StageError(f"{prefix} is non-finite")
    if isinstance(obj, dict):
        for key, value in obj.items():
            reject_nonfinite(value, f"{prefix}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            reject_nonfinite(value, f"{prefix}[{i}]")


def positive_int_bytes(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StageError(f"{name} must be a positive integer byte count, got {value!r}")
    return value
