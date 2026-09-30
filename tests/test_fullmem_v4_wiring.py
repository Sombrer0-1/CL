"""Wiring tests for board memory, executor recovery, and freeze identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion_repro.memory.observation import (
    RAW_RGB32_UINT8_BYTES,
    board_used_bytes,
    byte_space_initial_budgets,
    paper_data_representation,
    resolve_controller_memory_bytes,
)
from orion_repro.provenance import snapshot_source_tree
from orion_repro.stages.fullmem_v4.constants import STUDY_ID
from orion_repro.stages.fullmem_v4.executor import (
    _reconcile_boot,
    executor_once,
    health_check,
    infer_exit_code,
    reconcile_open_work,
)
from orion_repro.stages.fullmem_v4.queue import load_state, normalize_task, stage_paths
from orion_repro.stages.fullmem_v4.util import atomic_write_json, atomic_write_text


def test_board_observation_is_memtotal_minus_available():
    used = board_used_bytes(64 * 1024 * 1024, 10 * 1024 * 1024)
    assert used == 54 * 1024 * 1024
    assert resolve_controller_memory_bytes("board", board_used_bytes_value=used) == used
    assert resolve_controller_memory_bytes("device", gpu_peak_bytes=91 * 1024 * 1024) == 91 * 1024 * 1024
    with pytest.raises(ValueError):
        resolve_controller_memory_bytes("board", board_used_bytes_value=None)


def test_paper_byte_space_keeps_initial_counts():
    spec = paper_data_representation()
    assert spec["m_batch"] == RAW_RGB32_UINT8_BYTES
    mb0, mr0 = byte_space_initial_budgets(16, 200, m_batch=spec["m_batch"], m_frame=spec["m_frame"])
    assert mb0 / spec["m_batch"] == 16
    assert mr0 / spec["m_frame"] == 200


def test_source_snapshot_skips_executor_state_includes_design(tmp_path: Path):
    (tmp_path / "src" / "orion_repro").mkdir(parents=True)
    atomic_write_text(tmp_path / "src" / "orion_repro" / "x.py", "print(1)\n")
    state = tmp_path / "experiments" / "fullmem_v4" / "state"
    state.mkdir(parents=True)
    atomic_write_text(state / "executor.json", '{"heartbeat": 1}\n')
    design = tmp_path / "docs" / "fullmem_v4_design.md"
    design.parent.mkdir(parents=True)
    atomic_write_text(design, "# design\n")
    first = snapshot_source_tree(tmp_path)
    rels = {row["path"] for row in first["files"]}
    assert "docs/fullmem_v4_design.md" in rels
    assert "experiments/fullmem_v4/state/executor.json" not in rels
    atomic_write_text(state / "executor.json", '{"heartbeat": 2}\n')
    second = snapshot_source_tree(tmp_path)
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    atomic_write_text(design, "# design changed\n")
    third = snapshot_source_tree(tmp_path)
    assert third["aggregate_sha256"] != first["aggregate_sha256"]


def test_task_hash_binds_config_contents(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.constants.repo_root", lambda: tmp_path)
    (tmp_path / "experiments" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)
    cfg = tmp_path / "configs" / "fullmem_v4" / "probe.yaml"
    cfg.parent.mkdir(parents=True)
    atomic_write_text(cfg, "method_id: O-recon\ncontrolled_resource: device\n")
    task = {
        "study_id": STUDY_ID,
        "task_id": "hash-cfg",
        "kind": "dummy",
        "working_directory": str(tmp_path),
        "argv": ["python", "-m", "orion_repro.run", "--config", "configs/fullmem_v4/probe.yaml"],
    }
    first = normalize_task(task)
    atomic_write_text(cfg, "method_id: O-recon\ncontrolled_resource: board\n")
    second = normalize_task(task)
    assert first["content_hash"] != second["content_hash"]
    assert first["config_content_hashes"] != second["config_content_hashes"]


def test_reconcile_finalizes_dead_worker_before_inbox(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._unit_is_active", lambda unit: False)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.list_active_task_units", lambda: [])
    (tmp_path / "experiments" / STUDY_ID / "queue" / "running").mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "state").mkdir(parents=True)
    (tmp_path / "runs" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)
    atomic_write_json(
        tmp_path / "experiments" / STUDY_ID / "queue" / "running" / "g2-x.json",
        {"study_id": STUDY_ID, "task_id": "g2-x", "status": "running"},
    )
    decision = reconcile_open_work({"study_id": STUDY_ID, "mode": "running", "current_task_id": "g2-x"}, tmp_path)
    assert decision is not None
    assert decision["action"] == "finalize_dead"
    assert decision["task"]["task_id"] == "g2-x"


def test_reconcile_pauses_when_current_has_no_running_record(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._unit_is_active", lambda unit: False)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.list_active_task_units", lambda: [])
    (tmp_path / "experiments" / STUDY_ID / "queue" / "running").mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "state").mkdir(parents=True)
    (tmp_path / "runs" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)
    decision = reconcile_open_work({"study_id": STUDY_ID, "mode": "paused", "current_task_id": "g2-x"}, tmp_path)
    assert decision is not None
    assert decision["action"] == "pause"
    assert "stale_running_task" in decision["reason"]


def test_infer_exit_code_from_marker(tmp_path: Path):
    marker = tmp_path / "runs" / STUDY_ID / "g1-adopt-a.ok"
    marker.parent.mkdir(parents=True)
    marker.write_text("ok\n", encoding="utf-8")
    task = {
        "argv": ["python", "-m", "orion_repro.stages.fullmem_v4", "dummy", "--marker", "runs/fullmem_v4/g1-adopt-a.ok"],
        "working_directory": str(tmp_path),
    }
    assert infer_exit_code(task, tmp_path, "", code=None) == 0
    assert infer_exit_code({"argv": ["python"]}, tmp_path, '{"ok": true}\n', code=None) == 0
    assert infer_exit_code({"argv": ["python"]}, tmp_path, '{"ok": false, "reason": "cuda out of memory"}\n', code=None) is None


def test_health_check_rejects_active_task_unit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.list_active_task_units", lambda: ["orion-v4-task-g2-x.service"])
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._disk_ok", lambda root: (True, 10**12))
    (tmp_path / "experiments" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "queue" / "running").mkdir(parents=True)
    (tmp_path / "runs" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)
    ok, reason = health_check(tmp_path, {"swap_is_zero": True, "meminfo_kb": {"MemAvailable": 1000}})
    assert ok is False
    assert "task_units_active" in reason


def _wiring_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.queue.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.util.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.repo_root", lambda: tmp_path)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._unit_is_active", lambda unit: False)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.list_active_task_units", lambda: [])
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._pid_alive", lambda pid: False)
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor._disk_ok", lambda root: (True, 10**12))
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.acquire_locks", lambda paths: [])
    (tmp_path / "experiments" / STUDY_ID / "queue" / "inbox").mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "queue" / "running").mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "queue" / "done").mkdir(parents=True)
    (tmp_path / "experiments" / STUDY_ID / "state").mkdir(parents=True)
    (tmp_path / "runs" / STUDY_ID).mkdir(parents=True)
    (tmp_path / "reports" / STUDY_ID).mkdir(parents=True)


def test_reconcile_boot_open_work_is_oneshot(tmp_path: Path, monkeypatch):
    _wiring_root(tmp_path, monkeypatch)
    paths = stage_paths(tmp_path)
    atomic_write_json(
        paths["running"] / "g1-reboot-a.json",
        {"study_id": STUDY_ID, "task_id": "g1-reboot-a", "status": "running", "argv": ["python"]},
    )
    state = {
        "study_id": STUDY_ID,
        "mode": "running",
        "boot_id_at_batch_start": "boot-old",
        "current_task_id": "g1-reboot-a",
        "pause_reason": None,
        "completed_n": 0,
        "failed_n": 0,
    }
    identity = {"boot_id": "boot-new", "capacity": {"capacity_id": "mem16g"}}
    reason = _reconcile_boot(state, identity, tmp_path)
    assert reason == "host_reboot_with_open_work"
    assert load_state(tmp_path)["boot_id_at_batch_start"] == "boot-new"
    assert _reconcile_boot(load_state(tmp_path), identity, tmp_path) is None


def test_executor_once_reboot_finalizes_dead_keeps_inbox(tmp_path: Path, monkeypatch):
    _wiring_root(tmp_path, monkeypatch)
    paths = stage_paths(tmp_path)
    atomic_write_json(
        paths["running"] / "g1-reboot-a.json",
        {
            "study_id": STUDY_ID,
            "task_id": "g1-reboot-a",
            "status": "running",
            "kind": "dummy",
            "argv": ["python", "-m", "orion_repro.stages.fullmem_v4", "dummy", "--marker", "runs/fullmem_v4/g1-reboot-a.ok"],
            "working_directory": str(tmp_path),
        },
    )
    atomic_write_json(
        paths["inbox"] / "g1-reboot-b.json",
        {"study_id": STUDY_ID, "task_id": "g1-reboot-b", "status": "admitted", "seq": 801, "kind": "dummy", "argv": ["python"]},
    )
    atomic_write_json(
        paths["state"],
        {
            "study_id": STUDY_ID,
            "mode": "running",
            "boot_id_at_batch_start": "boot-old",
            "current_task_id": "g1-reboot-a",
            "worker_unit": "orion-v4-task-g1-reboot-a.service",
            "worker_pid": 1,
            "pause_reason": None,
            "completed_n": 0,
            "failed_n": 0,
        },
    )
    identity = {
        "boot_id": "boot-new",
        "capacity": {"capacity_id": "mem16g"},
        "swap_is_zero": True,
        "meminfo_kb": {"MemAvailable": 1000},
    }
    monkeypatch.setattr("orion_repro.stages.fullmem_v4.executor.collect_identity", lambda include_cuda=False: identity)
    first = executor_once(tmp_path)
    assert first["action"] == "paused"
    assert first["reason"] == "host_reboot_with_open_work"
    assert (paths["running"] / "g1-reboot-a.json").is_file()
    second = executor_once(tmp_path)
    assert second.get("adopted") is True
    assert second.get("kept_pause") is True
    state = load_state(tmp_path)
    assert state["mode"] == "paused"
    assert state.get("current_task_id") is None
    assert (paths["done"] / "g1-reboot-a.json").is_file()
    assert (paths["inbox"] / "g1-reboot-b.json").is_file()
    assert not (paths["running"] / "g1-reboot-a.json").exists()
