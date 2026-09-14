from pathlib import Path

import pytest

from orion_repro.benchmarks.factory import _build_endless, split_endless_train_indices
from orion_repro.runner.spec import RunSpec


def _base_spec(*, feedback: str, patch_size: int = 32, name: str = "endless_ic"):
    return {
        "dataset": {
            "name": name,
            "n_experiences": 4,
            "patch_size": patch_size,
            "scenario": "Classes",
        },
        "controller": {"feedback_source": feedback},
    }


def test_smoke_endless_spec_loads():
    spec = RunSpec.from_file(Path("configs/smoke_endless_ic.yaml"))
    assert spec.dataset_name == "endless_ic"
    assert spec.raw["dataset"]["patch_size"] == 32
    assert spec.raw["model"]["num_classes"] == 5
    assert spec.raw["controller"]["feedback_source"] == "official_test_seen"


def test_endless_rejects_unknown_feedback():
    with pytest.raises(ValueError, match="unhandled endless feedback_source"):
        _build_endless(
            _base_spec(feedback="heldout_train_control"),
            Path("."),
            "heldout_train_control",
        )


def test_endless_rejects_non_paper_patch_size():
    with pytest.raises(ValueError, match="patch_size"):
        _build_endless(_base_spec(feedback="official_test_seen", patch_size=64), Path("."), "official_test_seen")


def test_class_id_maps_endless_names():
    from orion_repro.benchmarks.factory import class_id
    assert class_id(3) == 3
    assert class_id("People") == 3
    assert class_id("BG") == 0


def test_legacy_frame_random_split_rejected():
    with pytest.raises(ValueError, match="retired"):
        split_endless_train_indices(100, experience_id=0)
