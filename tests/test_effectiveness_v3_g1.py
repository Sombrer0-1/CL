"""G1 runtime and stage-pipeline checks for effectiveness_v3."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orion_repro.control.ablation import applied_optimizer_mode, initial_optimizer_mode
from orion_repro.control.urge import scale_budget, select_optimizer_mode, urge_factors
from orion_repro.evaluation.evaluator import evaluate_domains, snapshot_rng
from orion_repro.memory.host_enforcement import HostEnvelope, probe_delegated_cgroup
from orion_repro.memory.phase_recorder import PhaseRecorder
from orion_repro.memory.probe import ResourceSampler, ResourceSnapshot
from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.prefetch.queue import BoundedPrefetcher, PrefetchItem
from orion_repro.runner.artifacts import RunArtifacts
from orion_repro.runner.spec import RunSpec, validate_mapping
from orion_repro.stages.effectiveness_v3.calibration import CalibrationError, calibrate, mid_quota_mib
from orion_repro.stages.effectiveness_v3.constants import SLOT_COUNTS, STUDY_ID, TOTAL_SLOTS
from orion_repro.stages.effectiveness_v3.matrix import planned_cells
from orion_repro.stages.effectiveness_v3.protocol import freeze
from orion_repro.stages.effectiveness_v3.report import validate_pairs
from orion_repro.stages.effectiveness_v3.util import StageError

ROOT = Path(__file__).resolve().parents[1]


def test_urge_rejects_nonfinite_and_keeps_sigmoid_direction():
    with pytest.raises(ValueError, match="finite"):
        urge_factors(
            float("nan"), 0.5, 1.0, 10.0,
            kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=1.0, m_max_mib=10.0,
        )
    with pytest.raises(ValueError, match="finite"):
        scale_budget(16.0, 0.1, float("inf"), 0.05)
    low_p = urge_factors(0.1, 0.5, 1.0, 10.0, kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=1.0, m_max_mib=10.0)
    high_p = urge_factors(0.9, 0.5, 1.0, 10.0, kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=1.0, m_max_mib=10.0)
    assert low_p["factor_p"] > high_p["factor_p"]
    slow = urge_factors(0.5, 0.5, 5.0, 10.0, kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=1.0, m_max_mib=10.0)
    fast = urge_factors(0.5, 0.5, 0.1, 10.0, kp=0.25, ks=0.25, kl=0.25, km=0.25, p_th=0.5, s_th=0.5, latency_th_s=1.0, m_max_mib=10.0)
    assert slow["factor_l"] > fast["factor_l"]
    assert select_optimizer_mode(0.05, 0.05, equal_uses_gt=True) == "default"


def test_fixed_advanced_keeps_raw_suggestion():
    assert applied_optimizer_mode({"plugin_policy": "fixed_advanced"}, "default") == "advanced"
    assert applied_optimizer_mode({"plugin_policy": "fixed_default"}, "advanced") == "default"
    spec = {"controller": {"plugin_policy": "fixed_advanced"}, "algorithm": {"optional_start_enabled": False}}
    assert initial_optimizer_mode(spec) == "advanced"
    spec["controller"]["plugin_policy"] = "adaptive"
    assert initial_optimizer_mode(spec) == "default"
    smoke = RunSpec.from_file(ROOT / "configs/smoke.yaml").raw
    smoke["controller"]["plugin_policy"] = "fixed_advanced"
    validate_mapping(smoke, require_provenance=False)


def test_sampler_old_token_cannot_write_new_phase_peaks():
    sampler = ResourceSampler(interval_s=0.2)
    t0 = sampler.begin_phase("training", 0)
    old = ResourceSnapshot(
        timestamp_s=0, monotonic_s=0, phase="training", experience_index=0,
        proc_rss_bytes=50_000_000, children_rss_bytes=1_000_000, proc_pss_bytes=None,
        system_available_bytes=None, swap_used_bytes=0, gpu_alloc_bytes=0, gpu_reserved_bytes=0,
        gpu_alloc_peak_bytes=0, gpu_reserved_peak_bytes=0, gpu_global_used_bytes=None, gpu_global_free_bytes=None,
    )
    sampler.ingest(old, t0)
    t1 = sampler.begin_phase("evaluation", 0)
    late_old = ResourceSnapshot(
        timestamp_s=0, monotonic_s=0, phase="training", experience_index=0,
        proc_rss_bytes=90_000_000, children_rss_bytes=0, proc_pss_bytes=None,
        system_available_bytes=None, swap_used_bytes=0, gpu_alloc_bytes=0, gpu_reserved_bytes=0,
        gpu_alloc_peak_bytes=0, gpu_reserved_peak_bytes=0, gpu_global_used_bytes=None, gpu_global_free_bytes=None,
    )
    sampler.ingest(late_old, t0)
    new_snap = ResourceSnapshot(
        timestamp_s=0, monotonic_s=0, phase="evaluation", experience_index=0,
        proc_rss_bytes=12_000_000, children_rss_bytes=0, proc_pss_bytes=None,
        system_available_bytes=None, swap_used_bytes=0, gpu_alloc_bytes=0, gpu_reserved_bytes=0,
        gpu_alloc_peak_bytes=0, gpu_reserved_peak_bytes=0, gpu_global_used_bytes=None, gpu_global_free_bytes=None,
    )
    sampler.ingest(new_snap, t1)
    assert sampler.peaks_for(t1).rss_peak_bytes == 12_000_000
    assert sampler.peaks_for(t0).rss_peak_bytes == 90_000_000


def test_phase_record_keeps_end_rss_separate_from_sampled_peak(tmp_path, monkeypatch):
    from orion_repro.memory import phase_recorder as pr

    monkeypatch.setattr(pr, "synchronize_gpu", lambda *_: None)
    monkeypatch.setattr(pr, "reset_gpu_peak", lambda *_: None)
    monkeypatch.setattr(
        pr,
        "snapshot",
        lambda *a, **k: SimpleNamespace(
            gpu_alloc_bytes=1, gpu_alloc_peak_bytes=2, gpu_reserved_bytes=3, gpu_reserved_peak_bytes=4,
            proc_rss_bytes=111, children_rss_bytes=0, proc_pss_bytes=None, swap_used_bytes=0,
        ),
    )
    sampler = ResourceSampler(interval_s=1.0)
    arts = RunArtifacts(tmp_path)
    rec = PhaseRecorder(arts, quota_bytes=10, reservation_getter=lambda: 0, sampler=sampler)
    rec.begin("training", 0)
    token = rec._open[("training", 0)][1]
    sampler.ingest(
        ResourceSnapshot(
            timestamp_s=0, monotonic_s=0, phase="training", experience_index=0,
            proc_rss_bytes=500, children_rss_bytes=7, proc_pss_bytes=None,
            system_available_bytes=None, swap_used_bytes=0, gpu_alloc_bytes=0, gpu_reserved_bytes=0,
            gpu_alloc_peak_bytes=0, gpu_reserved_peak_bytes=0, gpu_global_used_bytes=None, gpu_global_free_bytes=None,
        ),
        token,
    )
    row = rec.end("training", 0)
    arts.close()
    assert row.proc_rss_bytes == 111
    assert row.sampled_rss_peak_bytes == 500
    assert row.sampled_children_rss_peak_bytes == 7
    assert row.sample_count == 1


def test_prefetch_drops_stale_experience_and_resets_repeat_iter():
    pf = BoundedPrefetcher(max_depth=4)

    def producer():
        yield PrefetchItem(1, 1, 0, "old-exp")
        yield PrefetchItem(2, 1, 1, "ok")

    pf.start(producer)
    item = pf.get(config_version=1, experience_id=2, timeout=2.0)
    pf.close()
    assert item.batch == "ok"
    assert pf.dropped_stale >= 1

    class Loader:
        def __len__(self):
            return 2

        def __iter__(self):
            yield 1
            yield 2

    wrapped = PrefetchingDataLoader(Loader(), experience_id=0, config_version=0, max_depth=1, use_queue=False)
    assert list(wrapped) == [1, 2]
    assert list(wrapped) == [1, 2]
    assert wrapped.batches_consumed == 2


def test_host_envelope_writes_child_not_parent(tmp_path):
    (tmp_path / "cgroup.procs").write_text("0\n", encoding="utf-8")
    parent_limit = tmp_path / "memory.max"
    parent_limit.write_text("max", encoding="utf-8")
    probe = probe_delegated_cgroup(tmp_path)
    assert probe["available"] is True
    env = HostEnvelope.create(tmp_path, 4096, "max")
    assert env.path.parent == tmp_path
    assert env.path.name.startswith("orion_h_")
    assert (env.path / "memory.max").read_text(encoding="utf-8") == "4096"
    assert parent_limit.read_text(encoding="utf-8") == "max"
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resource_envelope_keeps_tensor_when_bytes_unchanged():
    from orion_repro.memory.resource_envelope import ResourceEnvelope

    device = torch.device("cuda:0")
    env = ResourceEnvelope(device)
    first = env.transition(0, 1024 * 1024)
    tensor = env._tensor
    second = env.transition(1, 1024 * 1024)
    assert first.status == "ok"
    assert second.notes == "unchanged"
    assert env._tensor is tensor
    env.close()



def _min_eval(dataset):
    rows = []
    for batch in (128, 32, 8):
        rows.append(
            {
                "role": "eval_batch",
                "dataset": dataset,
                "status": "completed",
                "feedback_source": "development_val_seen",
                "eval_batch": batch,
                "full_stream": False,
                "n_expected": 1,
                "n_trained": 1,
                "n_evaluated": 1,
            }
        )
    return rows


def _min_manifest(l_cal_trained=9):
    runs = []
    for dataset, n in (("core50_nc", 9), ("splitcifar100", 10)):
        runs.extend(_min_eval(dataset))
        runs.append(
            {
                "role": "quota_scan",
                "dataset": dataset,
                "status": "completed",
                "feedback_source": "development_val_seen",
                "quota_mib": 128,
                "train_reserved_peak_ratio": 0.88,
                "full_stream": True,
                "n_expected": n,
                "n_trained": n,
                "n_evaluated": n,
            }
        )
        runs.append(
            {
                "role": "q_loose_verify",
                "dataset": dataset,
                "status": "completed",
                "feedback_source": "development_val_seen",
                "quota_mib": 256,
                "full_stream": True,
                "n_expected": n,
                "n_trained": n,
                "n_evaluated": n,
            }
        )
        for i in range(2):
            runs.append(
                {
                    "role": "l_cal",
                    "dataset": dataset,
                    "status": "completed",
                    "feedback_source": "development_val_seen",
                    "learning_s_by_experience": [1.0] * n,
                    "full_stream": True,
                    "n_expected": n,
                    "n_trained": l_cal_trained if dataset == "core50_nc" else n,
                    "n_evaluated": n,
                    "config_id": f"l{dataset}{i}",
                }
            )
        for role in ("static_search_tight", "static_search_loose"):
            for batch, replay in ((16, 200), (16, 2000), (64, 200), (64, 2000), (256, 200), (256, 2000)):
                runs.append(
                    {
                        "role": role,
                        "dataset": dataset,
                        "status": "completed",
                        "feedback_source": "development_val_seen",
                        "new_batch": batch,
                        "replay_capacity": replay,
                        "p_diag": 0.5 if batch == 16 else 0.4,
                        "s_initial": 0.5,
                        "online_total_s": 8 if replay == 200 else 9,
                        "config_id": f"{role}_{dataset}_b{batch}_r{replay}",
                        "full_stream": True,
                        "n_expected": n,
                        "n_trained": n,
                        "n_evaluated": n,
                    }
                )
    runs.append(
        {
            "role": "control_2x2",
            "dataset": "core50_nc",
            "status": "completed",
            "feedback_source": "development_val_seen",
            "integer_config_changed": True,
            "full_stream": True,
            "n_expected": 9,
            "n_trained": 9,
            "n_evaluated": 9,
        }
    )
    for row in runs:
        row.setdefault("source_hash", "b" * 64)
    return {"study_id": STUDY_ID, "runs": runs}


def test_calibration_rejects_missing_feedback_and_partial_streams():
    with pytest.raises(CalibrationError):
        calibrate(
            {
                "study_id": STUDY_ID,
                "runs": [
                    {
                        "role": "l_cal",
                        "dataset": "core50_nc",
                        "status": "completed",
                        "feedback_source": "official_test_seen",
                    }
                ],
            }
        )
    with pytest.raises(CalibrationError, match="incomplete"):
        calibrate(_min_manifest(l_cal_trained=8))


def test_calibrate_two_datasets_and_mid_quota():
    result = calibrate(_min_manifest())
    assert result["datasets"]["core50_nc"]["eval_batch"] == 128
    assert result["datasets"]["splitcifar100"]["quota_tight_mib"] == 128
    assert result["datasets"]["core50_nc"]["quota_mid_mib"] == 192
    assert mid_quota_mib(176, 352) == 256
    with pytest.raises(CalibrationError):
        mid_quota_mib(160, 176)


def test_freeze_and_planned_cells_keep_153_denominator():
    calibration = calibrate(_min_manifest())
    calibration["io_prefetch"]["constructed"] = False
    calibration["host"] = {"status": "unavailable", "row": {"status": "unavailable"}}
    calibration["scenario_coverage"]["dynamic_reservation"] = "failed_to_construct"
    calibration["scenario_coverage"]["io_prefetch"] = "failed_to_construct"
    calibration["scenario_coverage"]["host"] = "unavailable"
    frozen = freeze(
        {"kind": "experiment_design", "study_id": STUDY_ID},
        calibration,
        {
            "design_hash": "a" * 64,
            "source_hash": "b" * 64,
            "environment_lock_sha256": "c" * 64,
            "environment_actual": {"torch": "2.11.0+cu128"},
            "platform": {"logical_device": "cuda:0", "gpus": []},
            "data_hashes": {
                "core50_nc_dev": "d1",
                "core50_nc_formal": "d2",
                "splitcifar100_dev": "d3",
                "splitcifar100_formal": "d4",
            },
        },
        revision="r1",
    )
    eligible, excluded, all_cells = planned_cells(frozen)
    assert len(all_cells) == TOTAL_SLOTS
    assert len(eligible) + len(excluded) == TOTAL_SLOTS
    counts = {key: 0 for key in SLOT_COUNTS}
    for cell in all_cells:
        counts[cell["group"]] += 1
    assert counts == SLOT_COUNTS
    assert any(cell["group"] == "E" for cell in excluded)
    assert any(cell["group"] == "H" for cell in excluded)
    assert frozen["s04_policy"]["group_c"] == "scenario_not_realized"
    assert frozen["q_mid_policy"]["role"] == "group_G_quota_mid_only"
    assert frozen["calibration_source_hash"] == "b" * 64


def test_calibrate_rejects_mixed_source_hash():
    manifest = _min_manifest()
    manifest["runs"][0]["source_hash"] = "a" * 64
    with pytest.raises(CalibrationError, match="mixed source_hash"):
        calibrate(manifest)


def test_freeze_rejects_source_hash_mismatch():
    calibration = calibrate(_min_manifest())
    with pytest.raises(CalibrationError, match="does not match current tree"):
        freeze(
            {"kind": "experiment_design", "study_id": STUDY_ID},
            calibration,
            {
                "design_hash": "a" * 64,
                "source_hash": "a" * 64,
                "environment_lock_sha256": "c" * 64,
                "environment_actual": {"torch": "2.11.0+cu128"},
                "platform": {"logical_device": "cuda:0", "gpus": []},
                "data_hashes": {
                    "core50_nc_dev": "d1",
                    "core50_nc_formal": "d2",
                    "splitcifar100_dev": "d3",
                    "splitcifar100_formal": "d4",
                },
            },
            revision="thor_r2",
        )


def test_quota_scan_continues_until_ratio_window():
    from orion_repro.stages.effectiveness_v3.development import _next_quota_scan, _verified_loose, empty_manifest

    evidence = empty_manifest("r1")
    evidence["runs"].append(
        {
            "role": "s0_profile",
            "dataset": "core50_nc",
            "status": "completed",
            "quota_mib": 8192,
            "train_reserved_peak_ratio": 4000 / 8192,
        }
    )
    quotas, blocked = _next_quota_scan(evidence, "core50_nc")
    assert blocked is None
    assert quotas[0] == 4000
    assert len(quotas) == 8
    evidence["runs"].extend(
        {
            "role": "quota_scan",
            "dataset": "core50_nc",
            "status": "completed",
            "quota_mib": q,
            "train_reserved_peak_ratio": 4000 / q,
        }
        for q in quotas
    )
    more, blocked = _next_quota_scan(evidence, "core50_nc")
    assert blocked is None
    assert more[0] == quotas[-1] + 16
    evidence["runs"].append(
        {
            "role": "q_loose_verify",
            "dataset": "core50_nc",
            "status": "completed",
            "quota_mib": 256,
        }
    )
    assert _verified_loose(evidence, "core50_nc", 128) == 256
    assert _verified_loose(evidence, "core50_nc", 200) is None
    left = {
        key: 1
        for key in (
            "dataset",
            "split_manifest",
            "model_name",
            "optimizer",
            "seed",
            "eval_batch",
            "precision",
            "quota_bytes",
            "source_hash",
            "frozen_hash",
        )
    }
    right = dict(left)
    validate_pairs(left, right)
    right["frozen_hash"] = "other"
    with pytest.raises(StageError, match="frozen"):
        validate_pairs(left, right)


def test_evaluation_restores_bn_and_rng():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    model.train()
    model(torch.randn(4, 3))
    running = model[0].running_mean.clone()
    ds = torch.utils.data.TensorDataset(torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1]))
    rng = snapshot_rng()
    evaluate_domains(model, [(0, ds)], device=torch.device("cpu"), batch_size=2)
    assert torch.equal(model[0].running_mean, running)
    assert model.training is True
    after = snapshot_rng()
    assert torch.equal(rng["torch"], after["torch"])


def test_optional_plugins_off_on_off_on_keep_state_and_base_replay():
    from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark
    from torch.utils.data import TensorDataset

    from orion_repro.runner.spec import load_yaml
    from orion_repro.strategies.builder import apply_runtime_config, build_optimizer, build_strategy
    from orion_repro.strategies.toggles import TogglePlugin

    spec = load_yaml(ROOT / "configs/smoke_max_a.yaml")
    spec["algorithm"]["optional_start_enabled"] = False
    spec["algorithm"]["patterns_per_exp"] = 2
    spec["training"].update(new_batch=2, replay_batch=2)
    spec["replay"]["capacity"] = 8
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    strategy = build_strategy(model, build_optimizer(model, spec), spec, device=torch.device("cpu"))
    ds = [TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([i, i, i, i])) for i in range(4)]
    bench = dataset_benchmark(ds, ds)
    gem = next(p for p in strategy.plugins if isinstance(p, TogglePlugin) and p.name == "gem")
    replay = next(p for p in strategy.plugins if type(p).__name__ == "ReplayPlugin")
    sizes = []
    for i, mode in enumerate(["default", "advanced", "default", "advanced"]):
        apply_runtime_config(strategy, new_batch=2, replay_capacity=8, replay_batch=2, optimizer_mode=mode)
        strategy.train(bench.train_stream[i], num_workers=0)
        sizes.append(len(gem.inner.memory_x))
    assert sizes[1] >= sizes[0]
    assert sizes[2] == sizes[1]
    assert sizes[3] >= sizes[2]
    assert replay.storage_policy is not None


def test_reuse_v4_includes_platform_and_changes_with_source(tmp_path, monkeypatch):
    from orion_repro.runner import reuse as reuse_mod

    (tmp_path / "requirements.lock.txt").write_text("locked")
    (tmp_path / "split.json").write_text("{}")
    src = tmp_path / "src/orion_repro"
    src.mkdir(parents=True)
    (src / "a.py").write_text("x=1")
    monkeypatch.setattr(reuse_mod, "platform_fingerprint", lambda: {"gpus": [{"uuid": "gpu-a"}]})
    monkeypatch.setattr(reuse_mod, "actual_training_packages", lambda: {"torch": "2.11.0+cu128"})
    spec = {"reuse_version": 4, "dataset": {"split_manifest": "split.json"}, "frozen_hash": "h1"}
    first = reuse_mod.reuse_identity(spec, tmp_path)
    (src / "a.py").write_text("x=2")
    assert reuse_mod.reuse_identity(spec, tmp_path) != first
    spec3 = {"reuse_version": 3, "dataset": {"split_manifest": "split.json"}}
    assert reuse_mod.reuse_identity(spec3, tmp_path) != first

