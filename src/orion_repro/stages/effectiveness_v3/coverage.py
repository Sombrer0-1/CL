"""S01–S08 construction audit from development/formal records."""

from __future__ import annotations

from typing import Any

from orion_repro.stages.effectiveness_v3.constants import STUDY_ID
from orion_repro.stages.effectiveness_v3.util import StageError


def audit_scenarios(records: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    if records.get("study_id") not in {STUDY_ID, None} and records.get("kind") not in {
        "calibration_result",
        "formal_attempts",
        "probe_manifest",
    }:
        raise StageError("scenario audit received a foreign study object")
    coverage = dict((records.get("scenario_coverage") or {}))
    scenarios = {}
    mapping = {
        "S01": coverage.get("S01") or coverage.get("control_identifiable") or "not_run",
        "S02": coverage.get("S02") or "not_run",
        "S03": coverage.get("S03") or "not_run",
        "S04": coverage.get("S04") or coverage.get("dynamic_reservation") or "not_run",
        "S05": coverage.get("S05") or "not_run",
        "S06": coverage.get("S06") or "not_run",
        "S07": coverage.get("S07") or coverage.get("io_prefetch") or "not_run",
        "S08": coverage.get("S08") or "not_run",
    }
    for scenario_id, construction in mapping.items():
        status = str(construction)
        if status in {"realized", "pending_formal"}:
            construction_status = "realized" if status == "realized" else "not_run"
        elif status in {"failed_to_construct", "unavailable", "not_run"}:
            construction_status = status
        else:
            construction_status = "not_run"
        scenarios[scenario_id] = {
            "scenario_id": scenario_id,
            "implementation_status": "pending",
            "construction_status": construction_status,
            "effectiveness_verdict": "未验证",
            "raw": status,
        }
    return {
        "study_id": STUDY_ID,
        "kind": "scenario_coverage",
        "scenarios": scenarios,
        "rules": rules or {},
    }
