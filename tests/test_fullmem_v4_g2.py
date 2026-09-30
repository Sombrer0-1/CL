"""G2 unit tests: alignment table, sigmoid saturation, inventory, train gate."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orion_repro.stages.fullmem_v4.constants import ORION_PYTHON, STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.g2_inventory import inventory
from orion_repro.stages.fullmem_v4.g2_physical import memory_sigmoid_table
from orion_repro.stages.fullmem_v4.queue import normalize_task
from orion_repro.stages.fullmem_v4.util import StageError


def test_alignment_csv_exists_and_does_not_replace_history():
    root = repo_root()
    v4 = root / "docs" / "fullmem_v4_alignment.csv"
    history = root / "docs" / "alignment.csv"
    assert v4.is_file()
    assert history.is_file()
    with v4.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"topic", "classification", "adopted_v4", "rationale"} <= set(rows[0].keys())
    topics = {row["topic"] for row in rows}
    assert "observation_M" in topics
    assert "no_model_upsizing" in topics
    assert "endless_protocol" in topics


def test_memory_sigmoid_saturates_with_small_slack():
    table = {row["slack_mib"]: row for row in memory_sigmoid_table(64293.0, km=0.25)}
    assert table[0]["factor_m"] == pytest.approx(0.5, abs=1e-6)
    assert table[32]["factor_m"] >= 0.99
    assert table[4096]["factor_m"] > table[32]["factor_m"]


def test_inventory_marks_missing_raw_not_ready(tmp_path: Path):
    payload = inventory(tmp_path)
    assert payload["streams"]["splitcifar10"]["ready"] is False
    assert payload["streams"]["endless_ic"]["ready"] is False
    assert payload["streams"]["splitcifar10"]["class_order_10exp"] == [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]


def test_g2_fullstream_yaml_sets_experience_limit():
    text = (repo_root() / "configs" / "fullmem_v4" / "g2_cifar100_s0.yaml").read_text(encoding="utf-8")
    assert "experience_limit: 10" in text
    assert "enforcement: observed_only" in text
    gem = (repo_root() / "configs" / "fullmem_v4" / "g2_cifar100_er_gem.yaml").read_text(encoding="utf-8")
    assert "optional_start_enabled: true" in gem
    assert "plugin_policy: fixed_advanced" in gem


def test_g2_remaining_stream_yamls():
    root = repo_root() / "configs" / "fullmem_v4"
    prefetch = (root / "g2_cifar100_prefetch.yaml").read_text(encoding="utf-8")
    assert "enabled: true" in prefetch
    assert "experience_limit: 10" in prefetch
    import yaml

    yaml.safe_load(prefetch)
    ni = (root / "g2_core50_ni_s0.yaml").read_text(encoding="utf-8")
    assert "experience_limit: 8" in ni
    assert "scenario: ni" in ni
    nic = (root / "g2_core50_nic_s0.yaml").read_text(encoding="utf-8")
    assert "experience_limit: 79" in nic
    assert "scenario: nicv2_79" in nic
    endless = (root / "g2_endless_ic_s0.yaml").read_text(encoding="utf-8")
    assert "experience_limit: 4" in endless
    assert "split_policy: class_ordered_holdout_embargo_v2" in endless
    assert "patch_size: 32" in endless
    ore = (root / "g2_cifar100_orecon.yaml").read_text(encoding="utf-8")
    assert "method_id: O-recon" in ore
    assert "enabled: true" in ore
    full = (root / "g2_cifar100_orecon_full.yaml").read_text(encoding="utf-8")
    assert "controlled_resource: board" in full
    assert "plugin_policy: adaptive" in full
    assert "optional_plugins: gem_ewc" in full
    assert "m_batch: 3072.0" in full
    assert "model: 17" in full
    gss = (root / "g2_cifar100_gss.yaml").read_text(encoding="utf-8")
    assert "base: gss" in gss
    oracle = list((root).glob("g2_oracle_cifar100_b*_r*.yaml"))
    assert len(oracle) == 42


def test_occupancy_run_list_covers_g2_streams():
    from orion_repro.stages.fullmem_v4.g2_occupancy import RUNS

    assert "fullmem_v4_g2_cifar100_s0" in RUNS
    assert "fullmem_v4_g2_core50_nic_s0" in RUNS
    assert "fullmem_v4_g2_endless_wc_s0" in RUNS


def test_train_kind_still_refused():
    with pytest.raises(StageError):
        normalize_task(
            {
                "study_id": STUDY_ID,
                "task_id": "formal-1",
                "kind": "train",
                "argv": [ORION_PYTHON, "-m", "orion_repro.run"],
            }
        )
