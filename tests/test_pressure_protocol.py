import copy
import json
from pathlib import Path

import pytest
from orion_repro.pressure_protocol import (
    CalibrationError,
    DesignError,
    calibrate,
    emit,
    freeze,
    interleave_methods,
    validate_design,
)
from orion_repro.runner.spec import load_yaml


def _candidate(role, batch, replay, status="completed", p=0.5, s=0.5, t=10.0, **extra):
    row = dict(
        role=role,
        config_id=f"{role}_b{batch}_r{replay}",
        config=f"{role}_b{batch}_r{replay}.yaml",
        status=status,
        feedback_source="development_val_seen",
        new_batch=batch,
        replay_capacity=replay,
        p_diag=p,
        s_initial=s,
        online_total_s=t,
        eval_batch=32,
        quota_mib=128,
        run_id=f"run_{role}_{batch}_{replay}",
    )
    row.update(extra)
    return row


def _probe():
    runs = [
        _candidate("d0_eval_batch", 16, 200, eval_batch=128, status="cuda_oom", failure_phase="evaluation"),
        _candidate("d0_eval_batch", 16, 200, eval_batch=32, status="completed"),
        _candidate("d0_eval_batch", 16, 200, eval_batch=8, status="completed"),
    ]
    for q, ratio, status in ((112, 0.99, "cuda_oom"), (128, 0.88, "completed"), (160, 0.70, "completed")):
        runs.append(
            _candidate(
                "quota_scan",
                16,
                200,
                status=status,
                quota_mib=q,
                train_reserved_peak_ratio=ratio,
                config_id=f"q{q}",
            )
        )
    runs.append(_candidate("q_loose_verify", 16, 200, quota_mib=256, status="completed"))
    for i in range(2):
        runs.append(
            _candidate(
                "l_cal",
                16,
                200,
                quota_mib=256,
                learning_s_by_experience=[1.0, 2.0, 3.0, 2.0, 1.5, 2.5, 2.0, 1.0, 2.0],
                config_id=f"lcal{i}",
            )
        )
    for role in ("static_search_tight", "static_search_loose", "static_search_dyn"):
        for batch, replay in ((16, 200), (16, 2000), (64, 200), (64, 2000), (256, 200), (256, 2000)):
            status = "cuda_oom" if batch == 256 and replay == 2000 else "completed"
            p = 0.9 if (batch, replay) == (16, 2000) else 0.4
            runs.append(_candidate(role, batch, replay, status=status, p=p, s=0.7, t=12 if replay == 2000 else 9))
    runs.append(_candidate("control_2x2", 16, 200, integer_config_changed=True, config_id="O11"))
    runs.append(
        _candidate(
            "dyn_reservation_probe",
            16,
            200,
            reserved_bytes_by_experience=[0, 0, 0, 8, 8, 8, 0, 0, 0],
            transition_first_ok=True,
            quota_mib=128,
        )
    )
    runs.append(_candidate("io_off", 16, 200, supply_wait_ratio=0.2))
    runs.append(_candidate("io_on", 16, 200, supply_wait_ratio=0.05))
    return {"study_id": "pressure_v2", "runs": runs}


def test_design_is_not_an_executable_matrix():
    with pytest.raises(DesignError):
        validate_design({"kind": "run_matrix", "study_id": "pressure_v2", "formal_count": 54})
    design = load_yaml(Path("experiments/pressure_v2/design.yaml"))
    validate_design(design)


def test_calibrate_rejects_official_feedback_and_missing_outcomes():
    probe = _probe()
    probe["runs"][8]["feedback_source"] = "official_test_seen"
    with pytest.raises(CalibrationError, match="non-development"):
        calibrate(probe)
    probe = _probe()
    probe["runs"][8]["status"] = None
    with pytest.raises(CalibrationError, match="missing outcome"):
        calibrate(probe)


def test_static_selection_keeps_failures_and_picks_feasible_winner():
    result = calibrate(_probe())
    assert result["eval_batch"] == 32
    assert result["q_tight_mib"] == 128
    assert result["q_loose_mib"] == 256
    assert result["static_selections"]["tight"]["new_batch"] == 16
    assert result["static_selections"]["tight"]["replay_capacity"] == 2000
    assert result["static_selections"]["tight"]["n_resource_failed"] == 1
    assert result["io_prefetch"]["constructed"] is True


def test_freeze_and_emit_are_exactly_54_and_hash_stable(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(".").resolve())
    calibration = calibrate(_probe())
    identity = {
        "dataset_hash": "d" * 64,
        "source_hash": "s" * 64,
        "design_hash": "g" * 64,
    }
    frozen = freeze(calibration, identity)
    again = freeze(copy.deepcopy(calibration), copy.deepcopy(identity))
    assert frozen["frozen_hash"] == again["frozen_hash"]
    assert len(frozen["run_order"]) == 54
    rels = emit(frozen, tmp_path)
    assert len(rels) == 54
    assert len(set(rels)) == 54
    matrix = load_yaml(tmp_path / "experiments/pressure_v2/formal.yaml")
    assert matrix["frozen_hash"] == frozen["frozen_hash"]
    o00 = load_yaml(tmp_path / "configs/pressure_v2/formal/tight/O00_s0.yaml")
    o11 = load_yaml(tmp_path / "configs/pressure_v2/formal/tight/O11_s0.yaml")
    r11 = load_yaml(tmp_path / "configs/pressure_v2/formal/tight/R11_s0.yaml")
    assert o00["algorithm"]["optional_plugins"] == "gem_ewc"
    assert o11["controller"]["thresholds"]["latency_s"] == calibration["l_cal_s"]
    assert r11["controller"]["plugin_policy"] == "fixed_default"
    s0 = load_yaml(tmp_path / "configs/pressure_v2/formal/tight/S0_s0.yaml")
    assert s0["algorithm"]["optional_plugins"] == "none"
    assert not s0["controller"]["enabled"]


def test_interleave_rotates_by_seed():
    order = interleave_methods(["A", "B", "C"], seeds=(0, 1, 2))
    assert [row["method"] for row in order if row["seed"] == 0] == ["A", "B", "C"]
    assert [row["method"] for row in order if row["seed"] == 1] == ["B", "C", "A"]
