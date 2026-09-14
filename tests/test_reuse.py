import json
from orion_repro.runner.reuse import find_completed, reuse_identity


def test_only_completed_identical_runs_reused(tmp_path):
    for name, status in [('a', 'completed'), ('b', 'cuda_oom'), ('c', 'interrupted')]:
        path = tmp_path/'runs'/name
        path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps({'run_id':name,'status':status,'reuse_identity':'id'}))
    assert find_completed(tmp_path, 'id')['run_id'] == 'a'
    assert find_completed(tmp_path, 'different') is None
    assert find_completed(tmp_path, None) is None


def test_reuse_requires_manifest_and_invalidates_code_data_config(tmp_path):
    assert reuse_identity({'dataset':{}}, tmp_path) is None
    (tmp_path/'requirements.lock.txt').write_text('locked')
    (tmp_path/'split.json').write_text('{}')
    spec={'dataset':{'split_manifest':'split.json'}, 'run_id':'a'}
    first=reuse_identity(spec, tmp_path)
    spec['run_id']='b'
    assert reuse_identity(spec,tmp_path)==first
    (tmp_path/'split.json').write_text('{"different":1}')
    assert reuse_identity(spec,tmp_path)!=first


def test_reuse_ignores_experiment_id_and_claim_ids(tmp_path):
    (tmp_path/'requirements.lock.txt').write_text('locked')
    (tmp_path/'split.json').write_text('{}')
    a = {'dataset': {'split_manifest': 'split.json'}, 'experiment_id': 'E02', 'claim_ids': ['E02']}
    b = {'dataset': {'split_manifest': 'split.json'}, 'experiment_id': 'E03', 'claim_ids': ['E03', 'C03']}
    assert reuse_identity(a, tmp_path) == reuse_identity(b, tmp_path)


def test_v2_ignores_unrelated_docs_and_configs_but_not_training_source(tmp_path):
    (tmp_path/'requirements.lock.txt').write_text('locked')
    (tmp_path/'split.json').write_text('{}')
    source=tmp_path/'src/orion_repro';source.mkdir(parents=True)
    (source/'trainer.py').write_text('version=1')
    spec={'reuse_version':2, 'dataset':{'split_manifest':'split.json'}, 'training':{'batch':16}}
    original=reuse_identity(spec,tmp_path)
    (tmp_path/'PLAN.md').write_text('updated')
    (tmp_path/'configs').mkdir();(tmp_path/'configs/other.yaml').write_text('unrelated: true')
    assert reuse_identity(spec,tmp_path)==original
    spec['training']['batch']=32
    assert reuse_identity(spec,tmp_path)!=original
    spec['training']['batch']=16
    (source/'trainer.py').write_text('version=2')
    assert reuse_identity(spec,tmp_path)!=original
