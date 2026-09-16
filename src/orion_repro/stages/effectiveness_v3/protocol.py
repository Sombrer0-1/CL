"""Immutable protocol freeze for effectiveness_v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orion_repro.stages.effectiveness_v3.calibration import CalibrationError
from orion_repro.stages.effectiveness_v3.constants import (
    FORMAL_PROTOCOL_ID,
    PREFERENCE_ORDERS,
    PRIMARY_DATASETS,
    SEEDS,
    STUDY_ID,
    TOTAL_SLOTS,
)
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.coverage import audit_scenarios
from orion_repro.stages.effectiveness_v3.identity import close_identity, source_identity
from orion_repro.stages.effectiveness_v3.schema import json_without_hash, validate_frozen
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json, canonical_hash, reject_nonfinite


def freeze(
    design: dict[str, Any],
    calibration: dict[str, Any],
    identity: dict[str, Any],
    *,
    revision: str,
) -> dict[str, Any]:
    if design.get("study_id") != STUDY_ID or calibration.get("study_id") != STUDY_ID:
        raise CalibrationError("freeze inputs must belong to effectiveness_v3")
    datasets = calibration.get("datasets") or {}
    for name in PRIMARY_DATASETS:
        block = datasets.get(name) or {}
        for key in ("eval_batch", "quota_tight_bytes", "quota_loose_bytes", "latency_cal_s", "static_selections"):
            if block.get(key) in (None, "TBD", "tbd"):
                raise CalibrationError(f"{name} missing {key}")
        if block.get("quota_mid_mib") is None:
            raise CalibrationError(f"{name} has no interior mid quota: {block.get('quota_mid_status')}")
    closed = close_identity(identity)
    coverage = audit_scenarios(calibration)
    frozen = {
        "kind": "frozen_protocol",
        "schema": "effectiveness_v3_frozen_v1",
        "study_id": STUDY_ID,
        "protocol_id": FORMAL_PROTOCOL_ID,
        "revision": revision,
        "design_hash": closed.get("design_hash"),
        "source_hash": closed.get("source_hash"),
        "environment_lock_sha256": closed.get("environment_lock_sha256"),
        "environment_actual": closed.get("environment_actual"),
        "platform": closed.get("platform"),
        "data_hashes": closed.get("data_hashes"),
        "identity_hash": closed.get("identity_hash"),
        "calibration_hash": calibration.get("calibration_hash") or canonical_hash(calibration),
        "datasets": calibration["datasets"],
        "thr0": calibration["thr0"],
        "delta": calibration["delta"],
        "alpha": calibration["alpha"],
        "beta": calibration["beta"],
        "initial_counts": {
            "new_batch": calibration["initial_new_batch"],
            "replay_capacity": calibration["initial_replay"],
        },
        "optional_plugins": calibration["optional_plugins"],
        "m_batch": calibration["m_batch"],
        "m_frame": calibration["m_frame"],
        "cost_model_note": calibration["cost_model_note"],
        "reserved_bytes_by_experience": [int(x) for x in calibration["reserved_bytes_by_experience"]],
        "dyn_quota_bytes": int(calibration["dyn_quota_bytes"]),
        "seeds": list(SEEDS),
        "scenario_coverage": calibration["scenario_coverage"],
        "scenario_audit": coverage,
        "io_prefetch": calibration["io_prefetch"],
        "host": calibration["host"],
        "o00_latency_s": calibration["o00_latency_s"],
        "o00_memory_mib": calibration["o00_memory_mib"],
        "decay_delta": calibration["decay_delta"],
        "preference_orders": PREFERENCE_ORDERS,
        "formal_slots": TOTAL_SLOTS,
        "source_runs": calibration.get("source_runs"),
    }
    reject_nonfinite({k: v for k, v in frozen.items() if k != "host"}, "frozen")
    frozen["frozen_hash"] = canonical_hash(json_without_hash(frozen))
    validate_frozen(frozen)
    return frozen


def write_frozen(frozen: dict[str, Any], context: StageContext) -> Path:
    path = context.revision_dir / "frozen_protocol.json"
    context.assert_output_path(path)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        import json

        old = json.loads(existing)
        if old.get("frozen_hash") != frozen["frozen_hash"]:
            raise StageError(f"refusing to overwrite different frozen protocol at {path}")
        return path
    atomic_write_json(path, frozen)
    return path


def load_or_build_identity(context: StageContext) -> dict[str, Any]:
    return source_identity(context.root, design_path=context.design_path)
