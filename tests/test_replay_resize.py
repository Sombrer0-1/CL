from types import SimpleNamespace

import torch
from avalanche.benchmarks.utils import as_classification_dataset
from avalanche.training.plugins import ReplayPlugin

from orion_repro.strategies.builder import (
    UnsupportedAdaptationError,
    apply_runtime_config,
    replay_occupancy,
    validate_algorithm_combo,
)
from orion_repro.runner.spec import load_yaml
from pathlib import Path


class _Tiny(torch.utils.data.Dataset):
    def __init__(self, n: int, label: int = 0) -> None:
        self.data = torch.zeros(n, 3, 8, 8)
        self.targets = [label] * n

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def _filled_replay(n: int = 10) -> tuple[object, ReplayPlugin]:
    plugin = ReplayPlugin(mem_size=n)
    ds = as_classification_dataset(_Tiny(n, 0))
    plugin.storage_policy.post_adapt(None, SimpleNamespace(dataset=ds))
    strategy = SimpleNamespace(
        plugins=[plugin],
        train_mb_size=4,
        _orion_base="er",
        _orion_version_holder={"config_version": 0},
    )
    return strategy, plugin


def test_replay_shrink_changes_occupancy():
    strategy, plugin = _filled_replay(10)
    assert len(plugin.storage_policy.buffer) == 10
    state = apply_runtime_config(
        strategy, new_batch=4, replay_capacity=3, replay_batch=4
    )
    assert state["replay_requested"] == 3
    assert state["replay_occupancy"] == 3
    assert len(plugin.storage_policy.buffer) == 3
    assert plugin.storage_policy.max_size == 3


def test_replay_shrink_to_zero():
    strategy, plugin = _filled_replay(8)
    apply_runtime_config(strategy, new_batch=4, replay_capacity=0, replay_batch=4)
    assert replay_occupancy(strategy)["replay_occupancy"] == 0


def test_replay_expand_does_not_restore_discarded():
    strategy, plugin = _filled_replay(10)
    apply_runtime_config(strategy, new_batch=4, replay_capacity=3, replay_batch=4)
    ids_after_shrink = list(range(len(plugin.storage_policy.buffer)))
    apply_runtime_config(strategy, new_batch=4, replay_capacity=10, replay_batch=4)
    occ = replay_occupancy(strategy)
    assert occ["replay_max_size"] == 10
    assert occ["replay_occupancy"] == 3
    assert len(plugin.storage_policy.buffer) == 3
    assert len(ids_after_shrink) == 3


def test_gem_optional_plugins_are_rejected():
    spec = load_yaml(Path("configs/smoke_gem.yaml"))
    spec["algorithm"]["optional_plugins"] = "gem_ewc"
    try:
        validate_algorithm_combo(spec)
        raise AssertionError("expected UnsupportedAdaptationError")
    except UnsupportedAdaptationError:
        pass


def test_gem_controller_is_allowed():
    spec = load_yaml(Path("configs/smoke_gem.yaml"))
    spec["controller"]["enabled"] = True
    validate_algorithm_combo(spec)


def test_gem_ewc_only_is_allowed():
    spec = load_yaml(Path("configs/smoke_gem.yaml"))
    spec["algorithm"]["optional_plugins"] = "ewc"
    validate_algorithm_combo(spec)


def test_agem_plus_gem_is_rejected():
    spec = load_yaml(Path("configs/smoke_agem.yaml"))
    spec["algorithm"]["optional_plugins"] = "gem"
    try:
        validate_algorithm_combo(spec)
        raise AssertionError("expected UnsupportedAdaptationError")
    except UnsupportedAdaptationError:
        pass
