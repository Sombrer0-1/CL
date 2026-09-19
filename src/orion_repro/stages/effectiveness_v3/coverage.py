"""S01–S08 construction audit from development/formal records.

Effectiveness verdicts stay 未验证 until a G4 pairing report fills them.
A single integer_config_changed flag is not a scenario realization proof.
"""

from __future__ import annotations

from typing import Any

from orion_repro.stages.effectiveness_v3.constants import STUDY_ID
from orion_repro.stages.effectiveness_v3.util import StageError


def _status(raw: Any, *, default: str = "not_run") -> str:
    status = str(raw if raw is not None else default)
    if status in {"realized", "pending_formal", "failed_to_construct", "unavailable", "not_run"}:
        return "realized" if status == "realized" else status if status != "pending_formal" else "not_run"
    return "not_run"


def audit_scenarios(records: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    if records.get("study_id") not in {STUDY_ID, None} and records.get("kind") not in {
        "calibration_result",
        "formal_attempts",
        "probe_manifest",
    }:
        raise StageError("scenario audit received a foreign study object")
    coverage = dict((records.get("scenario_coverage") or {}))
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
    scenarios = {}
    dyn_window = records.get("dyn_window") or {}
    io = records.get("io_prefetch") or {}
    host = records.get("host") or {}
    extras = {
        "S04": {
            "experience_windows": {
                "high": dyn_window.get("high_experiences"),
                "recovery": dyn_window.get("recovery_experiences"),
                "mutation": dyn_window.get("mutation_experience"),
            },
            "predicate_values": {
                "observable_feedback_window": dyn_window.get("observable_feedback_window"),
                "transition_first_ok": dyn_window.get("transition_first_ok"),
            },
            "limitations": [dyn_window["limitation"]] if dyn_window.get("limitation") else [],
            "evidence_run_ids": [dyn_window.get("run_id")] if dyn_window.get("run_id") else [],
        },
        "S07": {
            "predicate_values": {
                "wait_ratio": io.get("wait_ratio"),
                "cpu_percent_mean": io.get("cpu_percent_mean"),
                "constructed": io.get("constructed"),
            },
            "limitations": list(io.get("limitations") or []),
        },
        "S01": {
            "predicate_values": {"control_identifiable": records.get("control_identifiable")},
            "limitations": [
                "integer_config_changed is not by itself an effectiveness verdict",
            ],
        },
    }
    for scenario_id, construction in mapping.items():
        construction_status = _status(construction)
        item = {
            "scenario_id": scenario_id,
            "implementation_status": "pending",
            "construction_status": construction_status,
            "effectiveness_verdict": "未验证",
            "raw": construction,
            "actions_applied": [],
            "failure_category": None,
        }
        item.update(extras.get(scenario_id, {}))
        scenarios[scenario_id] = item
    if str(host.get("status") or coverage.get("host") or "unavailable") in {"unavailable", "failed_to_construct"}:
        scenarios.setdefault("S03", {})
    return {
        "study_id": STUDY_ID,
        "kind": "scenario_coverage",
        "scenarios": scenarios,
        "host": host.get("status") or coverage.get("host") or "unavailable",
        "rules": rules or {},
        "note": "construction_status is not an Orion effectiveness verdict",
    }
