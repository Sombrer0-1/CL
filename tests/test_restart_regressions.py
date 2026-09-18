"""Regressions found while designing effectiveness_v3 (no benchmark claims)."""
import builtins
from types import SimpleNamespace

import pytest
import torch

from orion_repro import envcheck
from orion_repro.memory import phase_recorder
from orion_repro.memory.phase_recorder import PhaseRecorder
from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.runner.artifacts import RunArtifacts


def test_failed_phase_records_its_own_peak_not_previous_phase(tmp_path, monkeypatch):
    peak = [11]
    def snap(*args, **kwargs):
        return SimpleNamespace(gpu_alloc_bytes=peak[0], gpu_alloc_peak_bytes=peak[0],
            gpu_reserved_bytes=peak[0] + 1, gpu_reserved_peak_bytes=peak[0] + 1,
            proc_rss_bytes=1, children_rss_bytes=0, proc_pss_bytes=None, swap_used_bytes=0)
    monkeypatch.setattr(phase_recorder, 'snapshot', snap)
    monkeypatch.setattr(phase_recorder, 'synchronize_gpu', lambda *_: None)
    monkeypatch.setattr(phase_recorder, 'reset_gpu_peak', lambda *_: None)
    arts = RunArtifacts(tmp_path)
    rec = PhaseRecorder(arts, quota_bytes=100, reservation_getter=lambda: 0)
    rec.begin('training', 0)
    rec.end('training', 0)
    rec.begin('evaluation', 0)
    peak[0] = 71
    failed = rec.end_failed('evaluation', 0)
    assert failed.allocated_peak_bytes == 71
    assert failed.notes == 'failed'
    assert rec.end_failed('training', 1) is None  # no fallback to a different phase/index
    arts.close()
    assert 'failed' in (tmp_path / 'phase_trace.csv').read_text()


def test_phase_overlap_rejected_before_peak_reset(tmp_path, monkeypatch):
    resets = []
    monkeypatch.setattr(phase_recorder, 'synchronize_gpu', lambda *_: None)
    monkeypatch.setattr(phase_recorder, 'reset_gpu_peak', lambda *_: resets.append(True))
    arts = RunArtifacts(tmp_path)
    rec = PhaseRecorder(arts, quota_bytes=None, reservation_getter=lambda: 0)
    rec.begin('training', 0)
    with pytest.raises(RuntimeError, match='overlap'):
        rec.begin('evaluation', 0)
    assert len(resets) == 1
    arts.close()


@pytest.mark.parametrize('queued', [False, True])
def test_fallback_loader_times_next_including_loading(tmp_path, monkeypatch, queued):
    # A virtual clock advances only inside data loading, not in observation/yield.
    from orion_repro.prefetch import dataloader
    clock = [0.0]
    monkeypatch.setattr(dataloader.time, 'perf_counter', lambda: clock[0])
    class Loader:
        def __iter__(self):
            for i in range(3):
                clock[0] += 2.0
                yield i
        def __len__(self):
            return 3
    wrapped = PrefetchingDataLoader(Loader(), experience_id=0, config_version=0,
                                   max_depth=1, use_queue=queued)
    assert list(wrapped) == [0, 1, 2]
    assert wrapped.digest()['produce_s'] == 6.0


def test_envcheck_rejects_noop_optimizer():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(SystemExit) as exc:
        envcheck.verify_training_step(model, opt, torch.ones(2, 2), torch.tensor([0, 0]))
    assert exc.value.code == 2


def test_envcheck_accepts_actual_finite_update():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    envcheck.verify_training_step(model, opt, torch.ones(2, 2), torch.tensor([0, 0]))


def test_envcheck_wheel_must_cover_device_arch_not_sm_120():
    assert envcheck.cuda_sm_tag((11, 0)) == "sm_110"
    assert envcheck.cuda_sm_tag((12, 0)) == "sm_120"
    thor_wheel = ["sm_80", "sm_90", "sm_100", "sm_110", "sm_120"]
    assert envcheck.wheel_covers_device(thor_wheel, (11, 0))
    assert envcheck.wheel_covers_device(thor_wheel, (12, 0))
    assert not envcheck.wheel_covers_device(["sm_120"], (11, 0))


def test_envcheck_parses_jetson_unavailable_smi_vram():
    assert envcheck.parse_smi_memory_mib("[N/A]") is None
    assert envcheck.parse_smi_memory_mib("Not Supported") is None
    assert envcheck.parse_smi_memory_mib("32607") == 32607


def test_envcheck_requires_avalanche_import(monkeypatch):
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'avalanche':
            raise ImportError('test missing Avalanche')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(SystemExit) as exc:
        envcheck.required_avalanche_version()
    assert exc.value.code == 2


@pytest.mark.parametrize('row', [
    {}, {'feedback_source': None},
    {'feedback_source': 'development_val_seen', 'controller_feedback_source': 'official_test_seen'},
])
def test_calibration_rejects_missing_or_conflicting_feedback(row):
    from orion_repro.pressure_protocol import _require_dev_feedback, CalibrationError
    with pytest.raises(CalibrationError):
        _require_dev_feedback(row)


@pytest.mark.parametrize('queued, expected', [(False, 0.6), (True, 0.2)])
def test_calibration_does_not_add_overlapping_producer_and_wait(monkeypatch, tmp_path, queued, expected):
    from orion_repro import pressure_calibrate as cal
    spec = {'budget': {}, 'training': {'new_batch': 16, 'eval_batch': 32},
            'replay': {'capacity': 200}, 'controller': {'feedback_source': 'development_val_seen'},
            'prefetch': {'enabled': queued}}
    monkeypatch.setattr(cal, 'load_yaml', lambda *_: spec)
    monkeypatch.setattr(cal, 'latest_run_for_config', lambda *_: tmp_path)
    monkeypatch.setattr(cal, 'read_csv', lambda p: [
        {'learning_s': 10, 'prefetch_produce_s': 6, 'prefetch_wait_s': 2}
    ] if p.name == 'experience_metrics.csv' else [])
    assert cal.collect_row('fake.yaml', 'io_on' if queued else 'io_off')['supply_wait_ratio'] == expected


def test_controller_final_experience_writes_complete_checkpoint(tmp_path, monkeypatch):
    from pathlib import Path
    from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark
    from torch.utils.data import TensorDataset
    from orion_repro.runner import loop
    from orion_repro.runner.spec import load_yaml
    spec = load_yaml(Path('configs/smoke.yaml'))
    spec['run_id'] = 'final_checkpoint_regression'
    spec['training'].update(device='cpu', new_batch=2, replay_batch=2, eval_batch=2)
    spec['dataset']['experience_limit'] = 1
    spec['controller']['enabled'] = True
    spec['measurement']['checkpoint_policy'] = 'experience_boundary'
    ds = TensorDataset(torch.ones(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    benchmark = dataset_benchmark([ds], [ds])
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    monkeypatch.setattr(loop, 'build_benchmark', lambda _: benchmark)
    monkeypatch.setattr(loop, 'build_model', lambda _: torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10)))
    result = loop.run_from_spec(spec)
    assert result['status'] == 'completed'
    ckpt_dir = tmp_path / 'runs' / spec['run_id'] / 'checkpoints'
    for name in ['latest.pt', 'experience_0.pt']:
        payload = torch.load(ckpt_dir / name, map_location='cpu', weights_only=False)
        assert payload['completed_experiences'] == 1
