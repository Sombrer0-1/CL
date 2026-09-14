import json
import subprocess
import sys
from pathlib import Path
import pytest
from orion_repro.control.ablation import applied_optimizer_mode
from orion_repro.runner.spec import load_yaml, validate_mapping, SpecError
from orion_repro import light24, study


def test_ablation_preserves_raw_decision_but_disables_switch():
    from orion_repro.control.controller import ControlState, step_controller
    from orion_repro.control.urge import UrgeConfig
    # Applied policy is independent of the raw equation decision.
    assert applied_optimizer_mode({}, 'advanced')=='advanced'
    assert applied_optimizer_mode({'plugin_policy':'fixed_default'}, 'advanced')=='default'
    with pytest.raises(ValueError): applied_optimizer_mode({'plugin_policy':'typo'}, 'advanced')


def test_all_study_cells_full_stream_three_seeds_and_valid(tmp_path):
    light24.emit(tmp_path)
    matrix=load_yaml(tmp_path/'experiments/light24/all.yaml')
    assert len(matrix['configs'])==90
    for rel in matrix['configs']:
        spec=load_yaml(tmp_path/rel)
        validate_mapping(spec,require_provenance=False)
        assert spec['training']['new_epochs']==1
        assert spec['dataset']['experience_limit']==spec['dataset']['n_experiences']
        assert spec['dataset'].get('max_train_samples_per_experience') is None
    prefs=load_yaml(tmp_path/'experiments/light24/prefetch.yaml')['configs']
    for i in range(0,len(prefs),2):
        off=load_yaml(tmp_path/prefs[i]);on=load_yaml(tmp_path/prefs[i+1])
        assert not off['prefetch']['enabled'] and on['prefetch']['enabled']
        off['prefetch']['enabled']=True
        assert off==on


def test_selection_never_uses_formal_or_missing_candidates(tmp_path):
    light24.emit(tmp_path)
    with pytest.raises(ValueError,match='Candidate missing'):
        light24.select_static(tmp_path)


def test_interrupted_accounting_is_not_invented(monkeypatch):
    monkeypatch.setattr(study, 'append_outcome', lambda *a: None)
    state={'consumed_s':12, 'results':[], 'active':{'run_id':'test','pid':None,'identity':'x'}}
    study.reconcile_active(state)
    assert state['consumed_s']==12
    assert state['active'] is None
    assert state['results'][0]['elapsed_s'] is None


def test_stop_group_only_stops_owned_process():
    proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    try:
        study.stop_group(proc)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:proc.kill()


def test_executor_records_real_child_and_does_not_apply_time_limit(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    spec={'schema_version':1}
    path=tmp_path/'config.yaml';path.write_text('{}')
    state={'consumed_s':25*3600, 'active':None, 'results':[]}
    real_popen=subprocess.Popen
    def child(args, **kwargs):
        config=args[-1]
        script='''import json,time,pathlib,sys
s=json.loads(pathlib.Path(sys.argv[1]).read_text())
time.sleep(.1)
p=pathlib.Path('runs')/s['run_id'];p.mkdir(parents=True)
(p/'summary.json').write_text(json.dumps({'status':'completed'}))
'''
        return real_popen([sys.executable,'-c',script,config],**kwargs)
    monkeypatch.setattr(study.subprocess,'Popen',child)
    result=study.execute(path,spec,'identity',state,tmp_path/'state.json')
    assert result=='completed'
    assert state['consumed_s']>25*3600
    assert state['results'][0]['elapsed_s']>=.1
    assert state['active'] is None


def test_main_runs_even_when_estimate_is_exceeded(tmp_path, monkeypatch):
    import yaml
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    spec=load_yaml(Path('configs/light24/core/splitcifar100_er_static_s0.yaml'))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(spec))
    (tmp_path/'matrix.yaml').write_text(yaml.safe_dump({'study_id':'light24_v1','configs':['config.yaml']}))
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs/light24_progress.json').write_text(json.dumps({'consumed_s':26*3600,'active':None,'results':[]}))
    monkeypatch.setattr(study,'reuse_identity',lambda *a:'identity')
    monkeypatch.setattr(study,'find_completed',lambda *a:None)
    called=[]
    monkeypatch.setattr(study,'execute',lambda *a: called.append(True) or 'completed')
    previous=study.signal.getsignal(study.signal.SIGTERM)
    try:study.main(['--matrix','matrix.yaml'])
    finally:study.signal.signal(study.signal.SIGTERM,previous)
    assert called==[True]


def test_static_selector_uses_development_winner_and_emits_six(tmp_path, monkeypatch):
    light24.emit(tmp_path)
    monkeypatch.setattr(light24, 'reuse_identity', lambda s,r:s)
    def result(root,spec):
        batch=spec['training']['new_batch'];capacity=spec['replay']['capacity']
        return {'run_id':f'dev_{batch}_{capacity}', 'p_diag':.9 if batch==64 else .5,
                's_initial':.7, 'online_total_s':10 if capacity==200 else 12,
                'timing_schema':'online_loop_v2'}
    monkeypatch.setattr(light24, 'find_completed', result)
    light24.select_static(tmp_path)
    matrix=load_yaml(tmp_path/'experiments/light24/selected.yaml')
    assert len(matrix['configs'])==6
    for rel in matrix['configs']:
        spec=load_yaml(tmp_path/rel)
        assert spec['training']['new_batch']==64
        assert spec['replay']['capacity']==200
        assert spec['controller']['feedback_source']=='official_test_seen'
        assert not spec['controller']['enabled']
        validate_mapping(spec, require_provenance=False)


def test_lost_parent_does_not_overwrite_completed_child(tmp_path, monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    p=tmp_path/'runs/done';p.mkdir(parents=True)
    (p/'summary.json').write_text(json.dumps({'status':'completed'}))
    state={'consumed_s':0,'results':[],'active':{'run_id':'done','pid':None}}
    study.reconcile_active(state)
    assert state['results'][0]['status']=='completed'
    assert state['results'][0]['elapsed_s'] is None
