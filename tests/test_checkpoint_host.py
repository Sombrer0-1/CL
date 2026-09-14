from pathlib import Path

import torch
from torch.nn import Linear
from torch.optim import SGD

from orion_repro.memory.host_enforcement import probe_host_enforcement
from orion_repro.runner.checkpoint import (
    apply_checkpoint,
    save_checkpoint,
    serialize_plugins,
    restore_plugins,
    _LabeledTensorDataset,
)
from orion_repro.strategies.agem_compat import AdaptiveAGEMPlugin
from orion_repro.strategies.gem_compat import AdaptiveGEMPlugin
from orion_repro.strategies.gss_compat import AdaptiveGSSPlugin


def test_gem_plugin_checkpoint_roundtrip(tmp_path: Path):
    model = Linear(4, 2)
    opt = SGD(model.parameters(), lr=0.01)
    plugin = AdaptiveGEMPlugin(4, 0.5)
    plugin.memory_x[0] = torch.arange(8, dtype=torch.float32).view(2, 4)
    plugin.memory_y[0] = torch.tensor([0, 1])
    plugin.memory_tid[0] = torch.zeros(2, dtype=torch.long)
    strategy = type("S", (), {})()
    strategy.model = model
    strategy.optimizer = opt
    strategy.train_mb_size = 8
    strategy.plugins = [plugin]
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        strategy=strategy,
        ctrl_state=None,
        completed_experiences=1,
        matrix=torch.zeros(2, 2).numpy(),
        correct_mat=torch.zeros(2, 2).numpy(),
        totals=torch.ones(2, 2).numpy(),
    )
    fresh = AdaptiveGEMPlugin(4, 0.5)
    strategy2 = type("S", (), {})()
    strategy2.model = Linear(4, 2)
    strategy2.optimizer = SGD(strategy2.model.parameters(), lr=0.01)
    strategy2.train_mb_size = 1
    strategy2.plugins = [fresh]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    apply_checkpoint(strategy2, payload, device=torch.device("cpu"))
    torch.testing.assert_close(fresh.memory_x[0], plugin.memory_x[0])
    torch.testing.assert_close(fresh.memory_y[0], plugin.memory_y[0])
    assert strategy2.train_mb_size == 8
    assert payload["completed_experiences"] == 1


def test_gss_serialize_restore_keeps_occupancy():
    plugin = AdaptiveGSSPlugin(mem_size=4, mem_strength=1, input_size=[3])
    plugin.ext_mem_list_x[:2] = torch.arange(6, dtype=torch.float32).view(2, 3)
    plugin.ext_mem_list_y[:2] = torch.tensor([1, 0])
    plugin.ext_mem_list_current_index = 2
    strategy = type("S", (), {})()
    strategy.plugins = [plugin]
    rows = serialize_plugins(strategy)
    other = AdaptiveGSSPlugin(mem_size=4, mem_strength=1, input_size=[3])
    strategy2 = type("S", (), {})()
    strategy2.plugins = [other]
    restore_plugins(strategy2, rows)
    assert other.ext_mem_list_current_index == 2
    torch.testing.assert_close(other.ext_mem_list_y[:2], torch.tensor([1, 0]))


def test_replay_buffer_groups_roundtrip():
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.training.storage_policy import ReservoirSamplingBuffer
    from avalanche.benchmarks.utils.utils import as_classification_dataset

    plugin = ReplayPlugin(mem_size=8)
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    y = torch.tensor([0, 1, 0])
    buf = ReservoirSamplingBuffer(4)
    buf.buffer = as_classification_dataset(_LabeledTensorDataset(x, y))
    buf._buffer_weights = torch.tensor([0.9, 0.3, 0.1])
    plugin.storage_policy.buffer_groups[0] = buf
    plugin.storage_policy._num_exps = 1
    plugin.storage_policy.max_size = 8
    strategy = type("S", (), {})()
    strategy.plugins = [plugin]
    rows = serialize_plugins(strategy)
    fresh = ReplayPlugin(mem_size=8)
    strategy2 = type("S", (), {})()
    strategy2.plugins = [fresh]
    restore_plugins(strategy2, rows)
    restored = fresh.storage_policy.buffer_groups[0]
    assert len(restored.buffer) == 3
    assert fresh.storage_policy._num_exps == 1
    ys = [int(sample[1]) for sample in restored.buffer]
    assert ys == [0, 1, 0]


def test_agem_buffers_roundtrip():
    plugin = AdaptiveAGEMPlugin(patterns_per_experience=4, sample_size=2)
    from avalanche.benchmarks.utils.utils import as_classification_dataset

    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    y = torch.tensor([1, 0])
    plugin.buffers = [as_classification_dataset(_LabeledTensorDataset(x, y))]
    strategy = type("S", (), {})()
    strategy.plugins = [plugin]
    rows = serialize_plugins(strategy)
    fresh = AdaptiveAGEMPlugin(patterns_per_experience=4, sample_size=2)
    strategy2 = type("S", (), {})()
    strategy2.plugins = [fresh]
    restore_plugins(strategy2, rows)
    assert len(fresh.buffers) == 1
    assert len(fresh.buffers[0]) == 2
    assert fresh.buffer_dataloader is not None


def test_host_enforcement_probe_does_not_claim_cgroup_without_write():
    probe = probe_host_enforcement()
    assert probe["host_enforced"] in {"available", "blocked"}
    assert "cgroup" in probe
    if not probe["cgroup"].get("writable"):
        assert probe["host_enforced"] == "blocked"
