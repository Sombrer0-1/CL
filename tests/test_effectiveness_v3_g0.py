"""G0: effectiveness_v3 stage isolation. No historical writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from orion_repro.stages.effectiveness_v3.constants import STUDY_ID, TOTAL_SLOTS
from orion_repro.stages.effectiveness_v3.context import StageContext, historical_protection_report, inspect_payload
from orion_repro.stages.effectiveness_v3.executor import dry_run_matrix, process_signature, signatures_match
from orion_repro.stages.effectiveness_v3.schema import load_design, validate_design
from orion_repro.stages.effectiveness_v3.util import StageError


ROOT = Path(__file__).resolve().parents[1]


def test_design_is_not_executable_and_has_153_slots():
    design = load_design(ROOT / "experiments/effectiveness_v3/design.yaml")
    assert design["executable"] is False
    assert design["formal_slots"] == TOTAL_SLOTS
    with pytest.raises(StageError):
        validate_design({**design, "executable": True})
    with pytest.raises(StageError):
        validate_design({**design, "study_id": "pressure_v2"})
    with pytest.raises(StageError):
        validate_design({**design, "configs": ["x.yaml"]})


def test_inspect_is_read_only(tmp_path, monkeypatch):
    light_mtime = (ROOT / "STATUS.md").stat().st_mtime
    payload = inspect_payload(StageContext(ROOT, revision="r1"))
    assert payload["writes"] is False
    assert payload["study_id"] == STUDY_ID
    assert payload["formal_slots"] == TOTAL_SLOTS
    assert payload["historical_protection"]["n_files"] > 0
    assert (ROOT / "STATUS.md").stat().st_mtime == light_mtime
    assert not (ROOT / "configs/pressure_v2").samefile(ROOT / "configs/effectiveness_v3") if (ROOT / "configs/effectiveness_v3").exists() else True


def test_context_rejects_historical_and_foreign_paths(tmp_path):
    ctx = StageContext(ROOT, revision="r1")
    with pytest.raises(StageError, match="historical"):
        ctx.assert_output_path(ROOT / "reports/pressure_v2/CLOSEOUT.md")
    with pytest.raises(StageError, match="historical"):
        ctx.assert_output_path(ROOT / "configs/light24/formal.yaml")
    with pytest.raises(StageError, match="allowlist"):
        ctx.assert_output_path(ROOT / "runs/some_old_run/summary.json")
    allowed = ctx.assert_output_path(ROOT / "runs/effectiveness_v3_20260101T000000Z_abcd1234")
    assert allowed.name.startswith("effectiveness_v3_")
    with pytest.raises(StageError):
        ctx.reject_foreign_study({"study_id": "pressure_v2"})


def test_cli_inspect_and_reject_old_matrix():
    from orion_repro.stages.effectiveness_v3.__main__ import build_parser, main

    args = build_parser().parse_args(["develop", "--revision", "r1", "--plan-only"])
    assert args.revision == "r1"
    assert args.plan_only is True
    assert main(["inspect"]) in {0, 1}
    assert main(["run", "--dry-run", "--matrix", "experiments/pressure_v2/formal.yaml"]) == 2


def test_dry_run_matrix_does_not_create_progress(tmp_path):
    ctx = StageContext(tmp_path, revision="r1")
    matrix = {"study_id": STUDY_ID, "configs": [], "eligible_n": 0, "excluded_n": 0, "design_slots": TOTAL_SLOTS}
    out = dry_run_matrix(matrix, ctx)
    assert out["writes"] is False
    assert not (tmp_path / "runs").exists()


def test_orion_python_prefers_miniconda_layout(tmp_path, monkeypatch):
    from orion_repro.stages.effectiveness_v3 import constants as const

    mini = tmp_path / "miniconda3" / "envs" / "orion" / "bin" / "python"
    mini.parent.mkdir(parents=True)
    mini.write_text("")
    monkeypatch.delenv("ORION_PY", raising=False)
    monkeypatch.delenv("ORION_PYTHON", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(const.Path, "home", classmethod(lambda cls: tmp_path))
    assert const.orion_python() == mini


def test_orion_python_ignores_archived_dot_conda_layout(tmp_path, monkeypatch):
    from orion_repro.stages.effectiveness_v3 import constants as const

    legacy = tmp_path / ".conda" / "envs" / "orion" / "bin" / "python"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("")
    monkeypatch.delenv("ORION_PY", raising=False)
    monkeypatch.delenv("ORION_PYTHON", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(const.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(const.sys, "executable", "/tmp/not-legacy-orion")
    assert const.orion_python() == Path("/tmp/not-legacy-orion")


def test_historical_protection_hashes_tracked_files():
    report = historical_protection_report(ROOT)
    assert report["n_files"] > 100
    assert "aggregate_sha256" in report


def test_pid_signature_mismatch_is_dead():
    live = process_signature(1)
    saved = {"pid": "1", "starttime": "not-the-real-starttime", "cmdline": "other"}
    assert signatures_match(saved, live) is False
    assert signatures_match(None, live) is False
