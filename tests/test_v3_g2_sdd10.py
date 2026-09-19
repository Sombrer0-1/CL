"""G2 SDD §10 planner, evidence closure, DYN window, and pairing tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion_repro.stages.effectiveness_v3 import development as dev
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.report import PAIR_CORE, collect, render, validate_pairs
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def late_stage(monkeypatch, missing):
    monkeypatch.setattr(dev, "_role_complete", lambda e, r, d, n: r != missing)
    monkeypatch.setattr(dev, "chosen_eval_batch", lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, "_chosen_tight", lambda e, d: 256)
    monkeypatch.setattr(dev, "_verified_loose", lambda e, d, q: 512)


def test_sensitivity_includes_plan_52_half_values_delta_and_mid_budget(monkeypatch):
    late_stage(monkeypatch, "sensitivity")
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [
            dict(role="l_cal", dataset="core50_nc", status="completed", learning_s_by_experience=[2, 4]),
        ],
    }
    batch = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    names = {p.spec["method_id"] for p in batch.probes}
    assert "alpha_half" in names
    assert "beta_half" in names
    assert "lr_double" in names
    assert "delta_half_life" in names
    assert "quota_mid" in names
    assert "quota_loose" in names
    by = {p.spec["method_id"]: p.spec for p in batch.probes}
    assert by["alpha_half"]["controller"]["updates"]["alpha"] == 0.05
    assert by["beta_half"]["controller"]["updates"]["beta"] == 0.1
    assert by["lr_double"]["training"]["optimizer"]["lr"] == 0.02
    assert by["delta_half_life"]["controller"]["delta"] == pytest.approx(0.08664339756999316)
    assert by["quota_mid"]["budget"]["limit_bytes"] == 384 * 1024**2
    assert by["quota_loose"]["budget"]["limit_bytes"] == 512 * 1024**2


def test_plugin_loose_uses_q_loose_and_fixed_advanced(monkeypatch):
    late_stage(monkeypatch, "plugin_loose")
    batch = dev.plan_probes({"study_id": "effectiveness_v3"}, {"study_id": "effectiveness_v3", "runs": []}, StageContext(ROOT))
    assert len(batch.probes) == 1
    spec = batch.probes[0].spec
    assert batch.probes[0].role == "plugin_loose"
    assert spec["dataset"]["name"] == "core50_nc"
    assert spec["controller"]["plugin_policy"] == "fixed_advanced"
    assert spec["algorithm"]["optional_start_enabled"] is True
    assert spec["algorithm"]["optional_plugins"] == "gem_ewc"
    assert spec["budget"]["limit_bytes"] == 512 * 1024**2


def test_dyn_min_positive_after_transition_oom(monkeypatch):
    real_complete = dev._role_complete

    def fake_complete(evidence, role, dataset, n):
        if role in {"dyn_reservation_probe", "dyn_min_positive", "static_search_dyn"}:
            return real_complete(evidence, role, dataset, n)
        return True

    monkeypatch.setattr(dev, "_role_complete", fake_complete)
    monkeypatch.setattr(dev, "chosen_eval_batch", lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, "_chosen_tight", lambda e, d: 256)
    monkeypatch.setattr(dev, "_verified_loose", lambda e, d, q: 512)
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [
            dict(
                role="dyn_reservation_probe",
                dataset="core50_nc",
                status="cuda_oom",
                failure_phase="resource_transition",
                quota_mib=256,
                reserved_bytes_by_experience=[0, 0, 0, 16 * 1024**2, 16 * 1024**2, 16 * 1024**2, 0, 0, 0],
                transition_first_ok=False,
            ),
            dict(role="l_cal", dataset="core50_nc", status="completed", learning_s_by_experience=[2, 4]),
        ],
    }
    batch = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    assert len(batch.probes) == 1
    assert batch.probes[0].role == "dyn_min_positive"
    assert batch.probes[0].spec["resource_envelope"]["reserved_bytes_by_experience"][3] == 4 * 1024**2


def test_dyn_min_positive_not_planned_when_first_window_ok(monkeypatch):
    late_stage(monkeypatch, "static_search_dyn")
    sequence = [0, 0, 0, 4194304, 4194304, 4194304, 0, 0, 0]
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [
            dict(
                role="dyn_reservation_probe",
                status="completed",
                quota_mib=256,
                reserved_bytes_by_experience=sequence,
                transition_first_ok=True,
            )
        ],
    }
    batch = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    assert len(batch.probes) == 6
    assert all(p.role == "static_search_dyn" for p in batch.probes)


def test_control_2x2_covers_cifar_after_nc(monkeypatch):
    real_complete = dev._role_complete

    def fake_complete(evidence, role, dataset, n):
        if role == "control_2x2" and dataset == "splitcifar100":
            return real_complete(evidence, role, dataset, n)
        return True

    monkeypatch.setattr(dev, "_role_complete", fake_complete)
    monkeypatch.setattr(dev, "chosen_eval_batch", lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, "_chosen_tight", lambda e, d: 256)
    monkeypatch.setattr(dev, "_verified_loose", lambda e, d, q: 512)
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [dict(role="l_cal", dataset="splitcifar100", status="completed", learning_s_by_experience=[1, 3])],
    }
    batch = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    assert {p.spec["method_id"] for p in batch.probes} == {"O00", "O10", "O01", "O11"}
    assert all(p.dataset == "splitcifar100" for p in batch.probes)
    assert batch.probes[1].spec["controller"]["thresholds"]["latency_s"] == 2.0


def test_evidence_closure_detects_hash_mismatch(tmp_path):
    from orion_repro.stages.effectiveness_v3.evidence import close_probe_row

    run_dir = tmp_path / "runs" / "runA"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"status": "completed", "n_experiences_trained": 1, "n_experiences_evaluated": 1}))
    (run_dir / "phase_trace.csv").write_text("phase,experience_index\ntraining,0\n")
    row = {
        "probe_id": "p",
        "role": "eval_batch",
        "dataset": "core50_nc",
        "run_id": "runA",
        "status": "completed",
        "n_trained": 1,
        "n_evaluated": 1,
        "feedback_source": "development_val_seen",
        "artifact_hashes": {"summary.json": "0" * 64},
    }
    result = close_probe_row(row, StageContext(tmp_path, revision="thor_r1"))
    assert result["ok"] is False
    assert any("hash mismatch" in m for m in result["mismatches"])


def test_factor_diagnostics_are_offline_and_do_not_claim_effectiveness(tmp_path):
    from orion_repro.control.urge import urge_factors
    from orion_repro.stages.effectiveness_v3.evidence import factor_diagnostics

    run_dir = tmp_path / "runs" / "runB"
    run_dir.mkdir(parents=True)
    (run_dir / "experience_metrics.csv").write_text(
        "experience_index,p_diag,s_initial,learning_s,memory_mib\n0,0.4,0.5,2.0,10\n"
    )
    (run_dir / "control_trace.jsonl").write_text("")
    (tmp_path / "experiments/effectiveness_v3/revisions/thor_r1").mkdir(parents=True)
    manifest = {
        "study_id": "effectiveness_v3",
        "runs": [
            {
                "role": "l_cal",
                "dataset": "core50_nc",
                "status": "completed",
                "run_id": "runB",
                "learning_s_by_experience": [2.0],
            }
        ],
    }
    payload = factor_diagnostics(manifest, StageContext(tmp_path, revision="thor_r1"))
    offline = payload["datasets"]["core50_nc"]["runs"][0]["experiences"][0]["offline_factors"]
    expected = urge_factors(0.4, 0.5, 2.0, 10.0, kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=2.0, m_max_mib=4096.0)
    assert offline["urge"] == pytest.approx(expected["urge"])
    assert "effectiveness" not in payload["note"].lower() or "not" in payload["note"].lower()


def test_report_writes_required_tables_without_picking_a_winner(tmp_path):
    context = StageContext(tmp_path, revision="thor_r1")
    context.ensure_revision_dirs()
    matrix = {
        "study_id": "effectiveness_v3",
        "design_slots": 153,
        "eligible_n": 2,
        "excluded_n": 151,
        "cells": [
            {
                "cell_id": "A/core50_nc/tight/O11/0/-",
                "group": "A",
                "dataset": "core50_nc",
                "method": "O11",
                "seed": 0,
                "variant": None,
            },
            {
                "cell_id": "A/core50_nc/tight/S0/0/-",
                "group": "A",
                "dataset": "core50_nc",
                "method": "S0",
                "seed": 0,
                "variant": None,
            },
        ],
    }
    progress = {
        "study_id": "effectiveness_v3",
        "results": [
            {"cell_id": "A/core50_nc/tight/O11/0/-", "status": "completed", "run_id": "r1", "elapsed_s": 10},
            {"cell_id": "A/core50_nc/tight/O11/0/-", "status": "completed", "run_id": "r1b", "elapsed_s": 8},
            {"cell_id": "A/core50_nc/tight/S0/0/-", "status": "cuda_oom", "run_id": "r2", "elapsed_s": 3},
        ],
    }
    atomic_write_json(context.progress_path, progress)
    bundle = collect(matrix, context)
    path = render(bundle, context, calibration={"study_id": "effectiveness_v3", "scenario_coverage": {}})
    assert path.is_file()
    for name in (
        "coverage.csv",
        "attempts.csv",
        "means.csv",
        "paired.csv",
        "phase_resources.csv",
        "control_events.csv",
        "plugin_activity.csv",
        "supply.csv",
        "search_cost.csv",
        "scenario_coverage.json",
        "selection.json",
        "RESULTS.md",
    ):
        assert (context.report_dir / name).is_file()
    attempts = (context.report_dir / "attempts.csv").read_text()
    assert attempts.count("A/core50_nc/tight/O11/0/-") == 2
    coverage = json.loads((context.report_dir / "scenario_coverage.json").read_text())
    assert coverage["scenarios"]["S01"]["effectiveness_verdict"] == "未验证"


def test_validate_pairs_includes_resource_sequence_and_source():
    left = {key: 1 for key in PAIR_CORE}
    right = dict(left)
    validate_pairs(left, right)
    right["resource_sequence"] = "[1]"
    with pytest.raises(StageError, match="resource_sequence"):
        validate_pairs(left, right)
    right = dict(left)
    right["source_hash"] = 2
    with pytest.raises(StageError, match="source"):
        validate_pairs(left, right)


def test_identity_recalibrate_skips_plugin_cost(monkeypatch):
    monkeypatch.setattr(dev, "chosen_eval_batch", lambda e: dict(core50_nc=128, splitcifar100=128))
    monkeypatch.setattr(dev, "_role_complete", lambda e, r, d, n: r in {"eval_batch", "s0_profile"})
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [
            dict(
                role="s0_profile",
                dataset="core50_nc",
                status="completed",
                quota_mib=8192,
                train_reserved_peak_ratio=152 / 8192,
            ),
            dict(
                role="s0_profile",
                dataset="splitcifar100",
                status="completed",
                quota_mib=8192,
                train_reserved_peak_ratio=152 / 8192,
            ),
        ],
    }
    batch = dev.plan_probes(
        {"study_id": "effectiveness_v3"},
        evidence,
        StageContext(ROOT),
        identity_recalibrate=True,
    )
    assert batch.probes
    assert {p.role for p in batch.probes} == {"quota_scan"}
    default = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    assert {p.role for p in default.probes} == {"plugin_cost"}


def test_identity_recalibrate_emits_s01_stamp_not_control(monkeypatch):
    monkeypatch.setattr(dev, "chosen_eval_batch", lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, "_chosen_tight", lambda e, d: 256)
    monkeypatch.setattr(dev, "_verified_loose", lambda e, d, q: 512)
    done = {
        "eval_batch",
        "s0_profile",
        "plugin_cost",
        "quota_scan",
        "q_loose_verify",
        "plugin_loose",
        "l_cal",
        "static_search_tight",
        "static_search_loose",
    }
    monkeypatch.setattr(dev, "_role_complete", lambda e, r, d, n: r in done)
    evidence = {
        "study_id": "effectiveness_v3",
        "runs": [dict(role="l_cal", dataset="core50_nc", status="completed", learning_s_by_experience=[2, 4])],
    }
    batch = dev.plan_probes(
        {"study_id": "effectiveness_v3"},
        evidence,
        StageContext(ROOT),
        identity_recalibrate=True,
    )
    assert len(batch.probes) == 1
    assert batch.probes[0].role == "identity_s01_stamp"
    assert batch.probes[0].spec["method_id"] == "O11"
    assert batch.probes[0].spec["budget"]["limit_bytes"] == 256 * 1024**2
    control = dev.plan_probes({"study_id": "effectiveness_v3"}, evidence, StageContext(ROOT))
    assert {p.role for p in control.probes} == {"control_2x2"}
