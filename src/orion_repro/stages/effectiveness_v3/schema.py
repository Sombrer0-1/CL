"""Design / probe / freeze / matrix schema checks for effectiveness_v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orion_repro.stages.effectiveness_v3.constants import (
    DESIGN_VERSION,
    FORMAL_PROTOCOL_ID,
    PRIMARY_DATASETS,
    SEEDS,
    SLOT_COUNTS,
    STUDY_ID,
    TOTAL_SLOTS,
)
from orion_repro.stages.effectiveness_v3.util import StageError, load_yaml, reject_nonfinite


REQUIRED_DESIGN = (
    "kind",
    "executable",
    "study_id",
    "design_version",
    "scenario_ids",
    "groups",
    "seeds",
    "calibration_rules",
    "parameter_sources",
    "gates",
)


def load_design(path: Path) -> dict[str, Any]:
    design = load_yaml(path)
    validate_design(design)
    return design


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_DESIGN if key not in design]
    if missing:
        raise StageError(f"design missing keys: {missing}")
    if design.get("kind") != "experiment_design":
        raise StageError("only kind=experiment_design is accepted")
    if design.get("study_id") != STUDY_ID:
        raise StageError("design study_id must be effectiveness_v3")
    if design.get("executable") is True:
        raise StageError("design files are not executable training matrices")
    if "configs" in design:
        raise StageError("design must not include a configs queue")
    if design.get("design_version") != DESIGN_VERSION:
        raise StageError(f"design_version must be {DESIGN_VERSION}")
    if design.get("formal_protocol_id") != FORMAL_PROTOCOL_ID:
        raise StageError("formal_protocol_id mismatch")
    if int(design.get("formal_slots", 0)) != TOTAL_SLOTS:
        raise StageError(f"formal_slots must be {TOTAL_SLOTS}")
    if list(design.get("seeds")) != list(SEEDS):
        raise StageError("seeds must be [0, 1, 2]")
    if list(design.get("primary_datasets")) != list(PRIMARY_DATASETS):
        raise StageError("primary_datasets must be core50_nc then splitcifar100")
    groups = design.get("groups") or {}
    for name, slots in SLOT_COUNTS.items():
        if name not in groups:
            raise StageError(f"design missing group {name}")
        if int(groups[name].get("slots", -1)) != slots:
            raise StageError(f"group {name} slots must be {slots}")
    reject_nonfinite(design, "design")
    return design


def validate_probe_row(row: dict[str, Any], *, require_feedback: bool = True) -> None:
    if row.get("status") is None:
        raise StageError(f"probe {row.get('probe_id') or row.get('config_id')} missing status")
    if require_feedback:
        sources = [row[key] for key in ("feedback_source", "controller_feedback_source") if key in row]
        if not sources or any(source != "development_val_seen" for source in sources):
            raise StageError(f"missing or non-development feedback rejected: {sources}")
    if row.get("status") == "implementation_error":
        raise StageError(f"implementation error in {row.get('probe_id') or row.get('config_id')}")


def validate_frozen(frozen: dict[str, Any]) -> dict[str, Any]:
    if frozen.get("kind") not in {"frozen_protocol", None} and frozen.get("schema") not in {
        "effectiveness_v3_frozen_v1",
        None,
    }:
        pass
    if frozen.get("study_id") != STUDY_ID:
        raise StageError("frozen study_id must be effectiveness_v3")
    if frozen.get("protocol_id") != FORMAL_PROTOCOL_ID:
        raise StageError("frozen protocol_id mismatch")
    if "frozen_hash" not in frozen:
        raise StageError("frozen_protocol missing frozen_hash")
    text = json_without_hash(frozen)
    for key, value in frozen.items():
        if key == "frozen_hash":
            continue
        if value is None:
            raise StageError(f"frozen field {key} is null")
        if isinstance(value, str) and value.strip().lower() in {"tbd", "todo", "placeholder"}:
            raise StageError(f"frozen field {key} is unresolved")
    reject_nonfinite({k: v for k, v in frozen.items() if k != "frozen_hash"}, "frozen")
    _ = text
    return frozen


def json_without_hash(frozen: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in frozen.items() if k != "frozen_hash"}
