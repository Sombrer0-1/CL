"""Regression tests for scientific contrasts damaged by migration-era scaffolding."""
from pathlib import Path
import pytest
from orion_repro.stages.effectiveness_v3 import development as dev
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.__main__ import build_parser

ROOT = Path(__file__).resolve().parents[1]


def late_stage(monkeypatch, missing):
    monkeypatch.setattr(dev, '_role_complete', lambda e, r, d, n: r != missing)
    monkeypatch.setattr(dev, 'chosen_eval_batch', lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, '_chosen_tight', lambda e, d: 256)
    monkeypatch.setattr(dev, '_verified_loose', lambda e, d, q: 512)


def test_control_factorial_preserves_memory_threshold_and_even_median(monkeypatch):
    late_stage(monkeypatch, 'control_2x2')
    evidence = {'study_id': 'effectiveness_v3', 'runs': [dict(role='l_cal', dataset='core50_nc', status='completed', learning_s_by_experience=[2, 4])]}
    batch = dev.plan_probes({'study_id': 'effectiveness_v3'}, evidence, StageContext(ROOT))
    rows = {p.spec['method_id']: p.spec for p in batch.probes}
    assert len(rows) == 4
    for method, latency, memory in [('O00',30,4096), ('O10',3,4096), ('O01',30,256), ('O11',3,256)]:
        assert rows[method]['controller']['thresholds']['m_max_mib'] == memory
        assert rows[method]['controller']['thresholds']['latency_s'] == latency
        assert rows[method]['budget']['limit_bytes'] == 256 * 1024**2


def test_dynamic_static_search_uses_actual_probe_schedule(monkeypatch):
    late_stage(monkeypatch, 'static_search_dyn')
    sequence = [0,0,0,4194304,4194304,4194304,0,0,0]
    evidence = {'study_id': 'effectiveness_v3', 'runs': [dict(role='dyn_reservation_probe', status='completed', quota_mib=256, reserved_bytes_by_experience=sequence)]}
    batch = dev.plan_probes({'study_id': 'effectiveness_v3'}, evidence, StageContext(ROOT))
    assert len(batch.probes) == 6
    assert all(p.spec['resource_envelope']['reserved_bytes_by_experience'] == sequence for p in batch.probes)
    assert all(not p.spec['controller']['enabled'] for p in batch.probes)


def test_plan_only_cannot_accidentally_execute():
    with pytest.raises(SystemExit):
        build_parser().parse_args(['develop', '--plan-only', '--execute'])


def test_static_search_rejects_missing_candidate_and_partial_success():
    from orion_repro.stages.effectiveness_v3.calibration import _select_static, CalibrationError
    from orion_repro.stages.effectiveness_v3.constants import STATIC_GRID
    rows = [dict(new_batch=b, replay_capacity=r, dataset='core50_nc', status='completed', n_expected=9, n_trained=9, n_evaluated=9, full_stream=True, p_diag=.5, s_initial=.5, online_total_s=10, config_id=str((b,r))) for b,r in STATIC_GRID]
    with pytest.raises(CalibrationError, match='six'):
        _select_static(rows[:-1])
    rows[0]['n_trained'] = 8
    with pytest.raises(CalibrationError, match='incomplete'):
        _select_static(rows)


def test_missing_review_blocks_freeze(tmp_path):
    from orion_repro.stages.effectiveness_v3.readiness import require_freeze_review
    from orion_repro.stages.effectiveness_v3.util import StageError
    with pytest.raises(StageError, match='G2 review missing'):
        require_freeze_review(StageContext(tmp_path, revision='thor_r1'))


def test_v3_source_identity_does_not_hash_its_own_outputs(tmp_path):
    from orion_repro.provenance import snapshot_source_tree
    source = tmp_path / 'src/orion_repro/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('x = 1')
    before = snapshot_source_tree(tmp_path, exclude_generated_v3=True)['aggregate_sha256']
    for rel in ('experiments/effectiveness_v3/revisions/thor_r1/g2_review.json', 'configs/effectiveness_v3/thor_r1/dev/probe.yaml'):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    assert snapshot_source_tree(tmp_path, exclude_generated_v3=True)['aggregate_sha256'] == before
    source.write_text('x = 2')
    assert snapshot_source_tree(tmp_path, exclude_generated_v3=True)['aggregate_sha256'] != before


def test_dyn_probe_uses_calibrated_latency(monkeypatch):
    late_stage(monkeypatch, 'dyn_reservation_probe')
    evidence = {
        'study_id': 'effectiveness_v3',
        'runs': [
            dict(role='l_cal', dataset='core50_nc', status='completed', learning_s_by_experience=[2, 4]),
            dict(role='s0_profile', dataset='core50_nc', status='completed', train_reserved_peak_ratio=0.5, quota_mib=256),
        ],
    }
    batch = dev.plan_probes({'study_id': 'effectiveness_v3'}, evidence, StageContext(ROOT))
    assert len(batch.probes) == 1
    thresholds = batch.probes[0].spec['controller']['thresholds']
    assert thresholds['latency_s'] == 3.0
    assert thresholds['m_max_mib'] == 256.0


def test_partial_control_wave_is_replanned(monkeypatch):
    real_complete = dev._role_complete

    def fake_complete(evidence, role, dataset, n):
        if role == 'control_2x2':
            return real_complete(evidence, role, dataset, n)
        return True

    monkeypatch.setattr(dev, '_role_complete', fake_complete)
    monkeypatch.setattr(dev, 'chosen_eval_batch', lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, '_chosen_tight', lambda e, d: 256)
    monkeypatch.setattr(dev, '_verified_loose', lambda e, d, q: 512)
    evidence = {
        'study_id': 'effectiveness_v3',
        'runs': [
            dict(role='control_2x2', dataset='core50_nc', status='completed', probe_id='control_O00_q256'),
            dict(role='control_2x2', dataset='core50_nc', status='completed', probe_id='control_O10_q256'),
            dict(role='control_2x2', dataset='core50_nc', status='completed', probe_id='control_O01_q256'),
            dict(role='control_2x2', dataset='core50_nc', status='not_run', probe_id='control_O11_q256'),
            dict(role='l_cal', dataset='core50_nc', status='completed', learning_s_by_experience=[2, 4]),
        ],
    }
    batch = dev.plan_probes({'study_id': 'effectiveness_v3'}, evidence, StageContext(ROOT))
    assert {p.spec['method_id'] for p in batch.probes} == {'O00', 'O10', 'O01', 'O11'}


def test_io_on_is_replanned_if_only_io_off_finished(monkeypatch):
    real_complete = dev._role_complete

    def fake_complete(evidence, role, dataset, n):
        if role in {'io_off', 'io_on'}:
            return real_complete(evidence, role, dataset, n)
        return True

    monkeypatch.setattr(dev, '_role_complete', fake_complete)
    monkeypatch.setattr(dev, 'chosen_eval_batch', lambda e: dict(core50_nc=32, splitcifar100=32))
    monkeypatch.setattr(dev, '_chosen_tight', lambda e, d: 256)
    monkeypatch.setattr(dev, '_verified_loose', lambda e, d, q: 512)
    evidence = {
        'study_id': 'effectiveness_v3',
        'runs': [
            dict(role='io_off', dataset='core50_nc', status='completed', probe_id='io_off_core50_nc'),
            dict(role='io_on', dataset='core50_nc', status='not_run', probe_id='io_on_core50_nc'),
        ],
    }
    batch = dev.plan_probes({'study_id': 'effectiveness_v3'}, evidence, StageContext(ROOT))
    assert {p.role for p in batch.probes} == {'io_off', 'io_on'}
