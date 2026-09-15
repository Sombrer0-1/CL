"""Resolved run spec load/validate (PLAN §11.2, T10)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

UNRESOLVED_STRINGS = {"auto", "tbd", "todo", "placeholder", "null"}

REQUIRED_TOP = [
    "schema_version",
    "phase",
    "experiment_id",
    "protocol_id",
    "method_id",
    "seeds",
    "model",
    "algorithm",
    "training",
    "replay",
    "controller",
    "budget",
    "prefetch",
    "measurement",
    "dataset",
]


class SpecError(ValueError):
    pass


def _walk_unresolved(obj: Any, prefix: str, errors: list[str]) -> None:
    if isinstance(obj, str) and obj.strip().lower() in UNRESOLVED_STRINGS:
        errors.append(f"{prefix} is unresolved string {obj!r}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_unresolved(v, f"{prefix}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_unresolved(v, f"{prefix}[{i}]", errors)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SpecError(f"{path} did not contain a mapping")
    return data


FORMAL_REQUIRED_AFTER_RESOLVE = (
    "dataset_manifest_sha256",
    "code_revision_or_snapshot",
    "environment_lock_sha256",
)


def validate_mapping(
    data: dict[str, Any],
    *,
    allow_null_keys: set[str] | None = None,
    require_provenance: bool | None = None,
) -> None:
    allow_null_keys = allow_null_keys or {
        "run_id",
        "dataset_manifest_sha256",
        "code_revision_or_snapshot",
        "environment_lock_sha256",
        "budget.limit_bytes",
        "budget.cost_model_path",
        "controller.thresholds.latency",
        "measurement.resume_checkpoint",
        "spec.run_id",
        "spec.dataset_manifest_sha256",
        "spec.code_revision_or_snapshot",
        "spec.environment_lock_sha256",
        "spec.budget.limit_bytes",
        "spec.budget.cost_model_path",
        "spec.controller.thresholds.latency",
        "spec.measurement.resume_checkpoint",
        "resource_envelope.schedule_path",
        "spec.resource_envelope.schedule_path",
        "data_supply.profile_path",
        "spec.data_supply.profile_path",
        "frozen_protocol_path",
        "spec.frozen_protocol_path",
    }
    if require_provenance is None:
        require_provenance = str(data.get("phase", "")) == "formal"
    if require_provenance:
        allow_null_keys = {k for k in allow_null_keys if k.split(".")[-1] not in FORMAL_REQUIRED_AFTER_RESOLVE}
    missing = [k for k in REQUIRED_TOP if k not in data]
    if missing:
        raise SpecError(f"missing keys: {missing}")
    errors: list[str] = []
    policy = str(data.get("measurement", {}).get("checkpoint_policy", "none"))
    if policy not in {"none", "experience_boundary"}:
        errors.append(
            "checkpoint_policy must be none or experience_boundary"
        )
    if data.get("controller", {}).get("plugin_policy", "adaptive") not in {"adaptive", "fixed_default"}:
        errors.append("controller.plugin_policy must be adaptive or fixed_default")
    _walk_unresolved(data, "spec", errors)

    def check_none(obj: Any, prefix: str) -> None:
        if obj is None and prefix not in allow_null_keys:
            errors.append(f"{prefix} is null without an allowed reason")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check_none(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_none(v, f"{prefix}[{i}]")

    check_none(data, "spec")
    if require_provenance:
        for key in FORMAL_REQUIRED_AFTER_RESOLVE:
            val = data.get(key)
            if not val:
                errors.append(f"formal spec missing {key}")
    if errors:
        raise SpecError("; ".join(errors))


def canonical_hash(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class RunSpec:
    raw: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "RunSpec":
        data = load_yaml(path)
        validate_mapping(data, require_provenance=False)
        return cls(raw=data, path=path)

    def resolved_copy(self, **updates: Any) -> dict[str, Any]:
        data = copy.deepcopy(self.raw)
        data.update(updates)
        return data

    @property
    def phase(self) -> str:
        return str(self.raw["phase"])

    @property
    def method_id(self) -> str:
        return str(self.raw["method_id"])

    @property
    def dataset_name(self) -> str:
        return str(self.raw["dataset"]["name"])
