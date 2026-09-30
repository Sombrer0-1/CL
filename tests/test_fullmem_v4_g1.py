"""G1 unit tests for fullmem_v4 queue, boot plan, and write guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion_repro.stages.fullmem_v4.boot import parse_extlinux, proposed_mem_label, proposed_mem64, rollback_to_primary
from orion_repro.stages.fullmem_v4.constants import ORION_PYTHON, STUDY_ID
from orion_repro.stages.fullmem_v4.executor import health_check, process_one
from orion_repro.stages.fullmem_v4.identity import capacity_matches, classify_capacity, parse_mem_cmdline
from orion_repro.stages.fullmem_v4.queue import list_inbox, load_state, normalize_task, stage_paths, submit
from orion_repro.stages.fullmem_v4.util import StageError, assert_write_path


SAMPLE_EXTLINUX = """TIMEOUT 30
DEFAULT primary

MENU TITLE L4T boot options

LABEL primary
      MENU LABEL primary kernel
      LINUX /boot/Image
      INITRD /boot/initrd
      APPEND ${cbootargs} root=PARTUUID=abc rw rootwait
"""


def test_parse_mem_cmdline_and_capacity():
    assert parse_mem_cmdline("root=/dev/sda rw") is None
    assert parse_mem_cmdline("root=/dev/sda mem=64G rw") == "64G"
    cap = classify_capacity("root=/dev/sda mem=64G", 64 * 1024 * 1024)
    assert cap["capacity_id"] == "mem64g"
    assert capacity_matches("any", {"capacity": cap})
    assert capacity_matches("mem64g", {"capacity": cap})
    assert not capacity_matches("mem16g", {"capacity": cap})


def test_boot_plan_adds_label_keeps_primary(tmp_path: Path):
    proposed = proposed_mem64(SAMPLE_EXTLINUX)
    parsed = parse_extlinux(proposed)
    assert parsed["default"] == "orion-mem64g"
    assert "primary" in parsed["labels"]
    assert "orion-mem64g" in parsed["labels"]
    assert "mem=64G" in parsed["labels"]["orion-mem64g"]["APPEND"]
    assert "mem=" not in parsed["labels"]["primary"]["APPEND"]
    rolled = rollback_to_primary(proposed)
    assert parse_extlinux(rolled)["default"] == "primary"
    assert "orion-mem64g" in parse_extlinux(rolled)["labels"]


def test_boot_plan_adds_32g_keeps_64g_and_primary():
    with_64 = proposed_mem64(SAMPLE_EXTLINUX)
    proposed = proposed_mem_label(
        with_64,
        mem="32G",
        label="orion-mem32g",
        menu="Orion fullmem_v4 mem=32G",
    )
    parsed = parse_extlinux(proposed)
    assert parsed["default"] == "orion-mem32g"
    assert "primary" in parsed["labels"]
    assert "orion-mem64g" in parsed["labels"]
    assert "orion-mem32g" in parsed["labels"]
    assert "mem=64G" in parsed["labels"]["orion-mem64g"]["APPEND"]
    assert "mem=32G" in parsed["labels"]["orion-mem32g"]["APPEND"]
    assert "mem=" not in parsed["labels"]["primary"]["APPEND"]


def test_classify_mem32():
    cap = classify_capacity("root=/dev/sda mem=32G", 32 * 1024 * 1024)
    assert cap["capacity_id"] == "mem32g"
    assert capacity_matches("mem32g", {"capacity": cap})
    assert not capacity_matches("mem64g", {"capacity": cap})


def test_boot_plan_adds_16g_keeps_32_64_and_primary():
    with_64 = proposed_mem64(SAMPLE_EXTLINUX)
    with_32 = proposed_mem_label(
        with_64,
        mem="32G",
        label="orion-mem32g",
        menu="Orion fullmem_v4 mem=32G",
    )
    proposed = proposed_mem_label(
        with_32,
        mem="16G",
        label="orion-mem16g",
        menu="Orion fullmem_v4 mem=16G",
    )
    parsed = parse_extlinux(proposed)
    assert parsed["default"] == "orion-mem16g"
    assert "primary" in parsed["labels"]
    assert "orion-mem64g" in parsed["labels"]
    assert "orion-mem32g" in parsed["labels"]
    assert "orion-mem16g" in parsed["labels"]
    assert "mem=64G" in parsed["labels"]["orion-mem64g"]["APPEND"]
    assert "mem=32G" in parsed["labels"]["orion-mem32g"]["APPEND"]
    assert "mem=16G" in parsed["labels"]["orion-mem16g"]["APPEND"]
    assert "mem=" not in parsed["labels"]["primary"]["APPEND"]


def test_boot_plan_adds_8g_keeps_16_32_64_and_primary():
    with_64 = proposed_mem64(SAMPLE_EXTLINUX)
    with_32 = proposed_mem_label(
        with_64,
        mem="32G",
        label="orion-mem32g",
        menu="Orion fullmem_v4 mem=32G",
    )
    with_16 = proposed_mem_label(
        with_32,
        mem="16G",
        label="orion-mem16g",
        menu="Orion fullmem_v4 mem=16G",
    )
    proposed = proposed_mem_label(
        with_16,
        mem="8G",
        label="orion-mem8g",
        menu="Orion fullmem_v4 mem=8G",
    )
    parsed = parse_extlinux(proposed)
    assert parsed["default"] == "orion-mem8g"
    assert "primary" in parsed["labels"]
    assert "orion-mem64g" in parsed["labels"]
    assert "orion-mem32g" in parsed["labels"]
    assert "orion-mem16g" in parsed["labels"]
    assert "orion-mem8g" in parsed["labels"]
    assert "mem=16G" in parsed["labels"]["orion-mem16g"]["APPEND"]
    assert "mem=8G" in parsed["labels"]["orion-mem8g"]["APPEND"]
    assert "mem=" not in parsed["labels"]["primary"]["APPEND"]


def test_classify_mem8():
    cap = classify_capacity("root=/dev/sda mem=8G", 8 * 1024 * 1024)
    assert cap["capacity_id"] == "mem8g"
    assert capacity_matches("mem8g", {"capacity": cap})
    assert not capacity_matches("mem16g", {"capacity": cap})


def test_boot_plan_refuses_existing_mem():
    dirty = SAMPLE_EXTLINUX.replace("rootwait", "rootwait mem=32G")
    with pytest.raises(StageError):
        proposed_mem64(dirty)


def test_submit_idempotent_and_conflict(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.constants.repo_root", lambda: tmp_path)
    (tmp_path / "experiments" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)
    task = {
        "study_id": STUDY_ID,
        "task_id": "g1-a",
        "seq": 1,
        "kind": "dummy",
        "argv": [ORION_PYTHON, "-m", "orion_repro.stages.fullmem_v4", "dummy", "--seconds", "1", "--marker", "runs/fullmem_v4/a.ok"],
        "required_capacity": "any",
        "batch_id": "detach-g1",
    }
    first = submit(task, root=tmp_path)
    second = submit(task, root=tmp_path)
    assert first["accepted"] and not first["idempotent"]
    assert second["idempotent"]
    other = dict(task)
    other["argv"] = list(task["argv"]) + ["--extra-ignored-should-conflict"]
    # extra unknown fields are dropped by normalize; change seq instead
    other = dict(task)
    other["seq"] = 9
    with pytest.raises(StageError):
        submit(other, root=tmp_path)


def test_normalize_refuses_train_until_g3():
    with pytest.raises(StageError):
        normalize_task(
            {
                "study_id": STUDY_ID,
                "task_id": "train-1",
                "kind": "train",
                "argv": [ORION_PYTHON, "-m", "orion_repro.run"],
            }
        )


def test_train_admitted_only_for_frozen_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    freeze_dir = tmp_path / "reports" / "fullmem_v4" / "g3"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "FREEZE.json").write_text(
        '{"admits_kind_train": true, "admitted_batches": ["g4-h4-mem8g"], "capacity_id": "mem8g"}\n',
        encoding="utf-8",
    )
    admitted = normalize_task(
        {
            "study_id": STUDY_ID,
            "task_id": "train-h4",
            "kind": "train",
            "argv": [ORION_PYTHON, "-c", "print(1)"],
            "required_capacity": "mem8g",
            "batch_id": "g4-h4-mem8g",
        }
    )
    assert admitted["kind"] == "train"
    with pytest.raises(StageError):
        normalize_task(
            {
                "study_id": STUDY_ID,
                "task_id": "train-h1",
                "kind": "train",
                "argv": [ORION_PYTHON, "-c", "print(1)"],
                "required_capacity": "mem8g",
                "batch_id": "g4-h1-mem8g",
            }
        )


def test_write_guard_blocks_v3(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    (tmp_path / "reports" / "effectiveness_v3").mkdir(parents=True)
    with pytest.raises(StageError):
        assert_write_path(tmp_path / "reports" / "effectiveness_v3" / "CLAIMS.md", tmp_path)
    allowed = assert_write_path(tmp_path / "reports" / "fullmem_v4" / "x.json", tmp_path)
    assert allowed.name == "x.json"


def _g1_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.constants.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.repo_root", lambda: tmp_path)
    (tmp_path / "experiments" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)


def test_process_one_pauses_on_disk_low_keeps_inbox(tmp_path: Path, monkeypatch):
    _g1_root(tmp_path, monkeypatch)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._disk_ok", lambda root: (False, 42))
    task = {
        "study_id": STUDY_ID,
        "task_id": "g1-diskfull-a",
        "seq": 810,
        "kind": "dummy",
        "argv": [ORION_PYTHON, "-m", "orion_repro.stages.fullmem_v4", "dummy", "--seconds", "1", "--marker", "runs/fullmem_v4/g1-diskfull-a.ok"],
        "required_capacity": "any",
        "batch_id": "g1-diskfull",
    }
    submit(task, root=tmp_path)
    inbox = list_inbox(tmp_path)
    identity = {"boot_id": "boot-1", "capacity": {"capacity_id": "mem16g"}}
    status = process_one(inbox[0], load_state(tmp_path), identity, tmp_path)
    state = load_state(tmp_path)
    paths = stage_paths(tmp_path)
    assert status == "paused"
    assert state["mode"] == "paused"
    assert str(state.get("pause_reason") or "").startswith("disk_low:")
    assert (paths["inbox"] / "g1-diskfull-a.json").is_file()
    assert not (paths["running"] / "g1-diskfull-a.json").exists()
    assert not (paths["done"] / "g1-diskfull-a.json").exists()
    assert state.get("current_task_id") is None


def test_health_check_reports_disk_low(tmp_path: Path, monkeypatch):
    _g1_root(tmp_path, monkeypatch)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._disk_ok", lambda root: (False, 7))
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.list_active_task_units", lambda: [])
    ok, reason = health_check(tmp_path, {"swap_is_zero": True, "meminfo_kb": {"MemAvailable": 1000}})
    assert ok is False
    assert reason.startswith("disk_low:")
