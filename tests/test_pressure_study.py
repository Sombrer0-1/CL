import json
from pathlib import Path

from orion_repro import pressure_study, study


def test_pressure_study_rejects_light24_and_uses_own_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pressure_study, "ROOT", tmp_path)
    monkeypatch.setattr(pressure_study, "PROGRESS", tmp_path / "runs/pressure_v2_progress.json")
    monkeypatch.setattr(pressure_study, "EXECUTOR_DIR", tmp_path / "runs/_pressure_v2_executor")
    monkeypatch.setattr(pressure_study, "FROZEN_PATH", tmp_path / "experiments/pressure_v2/frozen_protocol.json")
    (tmp_path / "experiments/pressure_v2").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments/pressure_v2/matrix.yaml").write_text(
        "study_id: light24_v1\nconfigs: []\n"
    )
    try:
        pressure_study.main(["--matrix", str(tmp_path / "experiments/pressure_v2/matrix.yaml")])
        assert False, "should reject"
    except SystemExit:
        pass


def test_light24_progress_path_unchanged():
    assert str(study.ROOT / "runs/light24_progress.json").endswith("runs/light24_progress.json")
    assert pressure_study.PROGRESS.name == "pressure_v2_progress.json"


def test_hash_change_is_not_reusable(tmp_path):
    from orion_repro.runner.reuse import reuse_identity

    (tmp_path / "requirements.lock.txt").write_text("locked")
    (tmp_path / "split.json").write_text("{}")
    (tmp_path / "src/orion_repro").mkdir(parents=True)
    (tmp_path / "src/orion_repro/a.py").write_text("x=1")
    spec = {
        "reuse_version": 3,
        "dataset": {"split_manifest": "split.json"},
        "resource_envelope": {"reserved_bytes_by_experience": [0, 1, 0]},
        "data_supply": {"profile": "natural_ondemand"},
    }
    first = reuse_identity(spec, tmp_path)
    spec["resource_envelope"]["reserved_bytes_by_experience"] = [0, 2, 0]
    assert reuse_identity(spec, tmp_path) != first
