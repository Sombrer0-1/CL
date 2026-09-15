from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from avalanche.training.plugins import ReplayPlugin
from orion_repro.runner.failures import failure_metadata
from orion_repro.runner.spec import load_yaml
from orion_repro.strategies.builder import build_model, build_optimizer, build_strategy, apply_runtime_config, UnsupportedAdaptationError
from orion_repro.strategies.toggles import TogglePlugin


def test_advanced_without_plugins_fails_before_any_mutation():
    replay=ReplayPlugin(mem_size=200)
    strategy=SimpleNamespace(plugins=[replay], train_mb_size=16)
    with pytest.raises(UnsupportedAdaptationError, match='requires installed'):
        apply_runtime_config(strategy,new_batch=32,replay_capacity=100,replay_batch=32,optimizer_mode='advanced')
    assert strategy.train_mb_size==16 and replay.mem_size==200


def test_explicit_real_plugins_switch_and_report_actual_state():
    spec=load_yaml(Path('configs/smoke_max_a.yaml'))
    spec['algorithm']['optional_start_enabled']=False
    model=build_model(spec)
    strategy=build_strategy(model,build_optimizer(model,spec),spec,device=torch.device('cpu'))
    toggles=[p for p in strategy.plugins if isinstance(p,TogglePlugin)]
    assert {type(p.inner).__name__ for p in toggles}=={'SparseGEMPlugin','EWCPlugin'}
    for mode,enabled in [('advanced',True),('default',False)]:
        state=apply_runtime_config(strategy,new_batch=16,replay_capacity=200,replay_batch=16,optimizer_mode=mode)
        assert state['applied_optional_plugins']=={'gem':enabled,'ewc':enabled}
        assert all(p.enabled==enabled for p in toggles)


def test_evaluation_oom_is_not_training_start_failure():
    spec=load_yaml(Path('configs/smoke.yaml'))
    meta=failure_metadata(spec,phase='evaluation',experience=0,trained=1,evaluated=0)
    assert meta['failure_phase']=='evaluation'
    assert meta['n_experiences_trained']==1 and meta['n_experiences_run']==0
    assert meta['eval_batch']==128


def test_real_training_populates_plugin_state_then_disabled_hooks_pause():
    from avalanche.benchmarks.scenarios.deprecated.generators import dataset_benchmark
    from torch.utils.data import TensorDataset
    spec=load_yaml(Path('configs/smoke_max_a.yaml'))
    spec['algorithm']['optional_start_enabled']=False
    spec['algorithm']['patterns_per_exp']=2
    spec['training'].update(new_batch=2,replay_batch=2)
    spec['replay']['capacity']=4
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,10))
    strategy=build_strategy(model,build_optimizer(model,spec),spec,device=torch.device('cpu'))
    ds=[TensorDataset(torch.randn(4,3,32,32),torch.tensor([i,i,i,i])) for i in [0,1]]
    bench=dataset_benchmark(ds,ds)
    apply_runtime_config(strategy,new_batch=2,replay_capacity=4,replay_batch=2,optimizer_mode='advanced')
    strategy.train(bench.train_stream[0],num_workers=0)
    plugins={p.name:p for p in strategy.plugins if isinstance(p,TogglePlugin)}
    assert len(plugins['gem'].inner.memory_x)>0
    assert len(plugins['ewc'].inner.importances)>0
    lengths=(len(plugins['gem'].inner.memory_x),len(plugins['ewc'].inner.importances))
    apply_runtime_config(strategy,new_batch=2,replay_capacity=4,replay_batch=2,optimizer_mode='default')
    strategy.train(bench.train_stream[1],num_workers=0)
    assert lengths==(len(plugins['gem'].inner.memory_x),len(plugins['ewc'].inner.importances))
