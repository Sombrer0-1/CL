from pathlib import Path

from orion_repro.runner.spec import RunSpec


def test_smoke_spec_loads():
    spec = RunSpec.from_file(Path("configs/smoke.yaml"))
    assert spec.phase == "smoke"
    assert spec.method_id == "er_static"
    assert spec.raw["training"]["new_epochs"] == 1
    assert spec.raw["dataset"]["n_experiences"] == 10


def test_formal_template_loads_with_null_provenance():
    spec = RunSpec.from_file(Path("configs/formal/e02/splitcifar10/er_static/seed0.yaml"))
    assert spec.phase == "formal"
    assert spec.raw["dataset_manifest_sha256"] is None
    assert spec.raw["controller"]["feedback_source"] == "official_test_seen"
