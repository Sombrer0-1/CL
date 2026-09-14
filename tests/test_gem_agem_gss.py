"""GEM/AGEM/GSS properties: paper constraints, official selection, resize, toggles."""

from types import SimpleNamespace

import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD

from orion_repro.strategies.agem_compat import AdaptiveAGEMPlugin
from orion_repro.strategies.builder import apply_runtime_config, build_model, build_optimizer, build_strategy
from orion_repro.strategies.capacity import patterns_per_experience
from orion_repro.strategies.gem_compat import AdaptiveGEMPlugin
from orion_repro.strategies.gss_compat import AdaptiveGSSPlugin
from orion_repro.strategies.toggles import TogglePlugin
from orion_repro.runner.spec import load_yaml
from pathlib import Path


def _flat_grad(model) -> torch.Tensor:
    return torch.cat(
        [
            p.grad.detach().flatten()
            if p.grad is not None
            else torch.zeros(p.numel())
            for p in model.parameters()
        ]
    )


def test_patterns_per_experience_mapping():
    assert patterns_per_experience(200, 10) == 20
    assert patterns_per_experience(0, 9) == 0
    assert patterns_per_experience(50, 9) == 5


def test_gem_projection_satisfies_memory_dot_constraint():
    """Lopez-Paz GEM: projected gradient should not decrease memory losses.

    Paper constraint is g_k · g' >= 0. Avalanche QP uses memory_strength as
    slack, so the observable bound is g_k · g' >= -γ - eps.
    """
    torch.manual_seed(0)
    model = Linear(4, 2, bias=False)
    opt = SGD(model.parameters(), lr=0.1)
    plugin = AdaptiveGEMPlugin(patterns_per_experience=2, memory_strength=0.5)
    plugin.memory_x[0] = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    plugin.memory_y[0] = torch.tensor([0, 1])
    plugin.memory_tid[0] = torch.zeros(2, dtype=torch.long)
    strategy = SimpleNamespace(
        clock=SimpleNamespace(train_exp_counter=1),
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        _criterion=CrossEntropyLoss(),
    )
    plugin.before_training_iteration(strategy)
    assert plugin.G.ndim == 2 and plugin.G.shape[0] == 1
    g_mem = plugin.G[0]
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = (-g_mem[offset : offset + n]).view_as(p).clone()
        offset += n
    assert (torch.mv(plugin.G, _flat_grad(model)) < 0).any()
    plugin.after_backward(strategy)
    dots = torch.mv(plugin.G, _flat_grad(model))
    assert torch.all(dots >= -plugin.memory_strength - 1e-3)
    assert plugin.projection_count == 1
    assert plugin.auxiliary_visits == 2


def test_agem_projection_makes_reference_dot_nonnegative():
    """Chaudhry A-GEM: if g·g_ref < 0, the projected gradient is orthogonal to g_ref."""
    model = Linear(3, 1, bias=False)
    plugin = AdaptiveAGEMPlugin(patterns_per_experience=4, sample_size=2)
    plugin.buffers = [object()]
    plugin.reference_gradients = torch.tensor([1.0, 0.0, 0.0])
    model.weight.grad = torch.tensor([[-1.0, 0.4, -0.2]])
    before = model.weight.grad.view(-1)
    assert torch.dot(before, plugin.reference_gradients) < 0
    plugin.after_backward(SimpleNamespace(model=model))
    after = model.weight.grad.view(-1)
    assert torch.dot(after, plugin.reference_gradients) >= -1e-5
    assert plugin.projection_count == 1


class _MbStrategy:
    """Avalanche Naive exposes mb_x via mbatch; it has no mb_x setter."""

    def __init__(self, model, criterion, batch_x, batch_y):
        self.model = model
        self.device = torch.device("cpu")
        self._criterion = criterion
        self.mbatch = [batch_x, batch_y]

    @property
    def mb_x(self):
        return self.mbatch[0]

    @property
    def mb_y(self):
        return self.mbatch[1]


def test_gss_after_forward_fills_then_scores_when_full():
    """GSS must invoke official scoring once the buffer is full; occupancy is capped."""
    torch.manual_seed(0)
    plugin = AdaptiveGSSPlugin(mem_size=4, mem_strength=1, input_size=[4])
    model = Linear(4, 2)
    criterion = CrossEntropyLoss()

    def _step(batch_x, batch_y):
        strategy = _MbStrategy(model, criterion, batch_x, batch_y)
        plugin.device = torch.device("cpu")
        plugin.after_forward(strategy)

    _step(torch.randn(2, 4), torch.tensor([0, 1]))
    assert plugin.ext_mem_list_current_index == 2
    assert plugin.selection_calls == 1
    _step(torch.randn(2, 4), torch.tensor([0, 1]))
    assert plugin.ext_mem_list_current_index == 4
    full_calls_before = plugin.full_buffer_score_calls
    _step(torch.randn(2, 4), torch.tensor([1, 0]))
    assert plugin.ext_mem_list_current_index == 4
    assert plugin.full_buffer_score_calls == full_calls_before + 1
    assert plugin.mem_size == 4


def test_gss_full_buffer_when_batch_exceeds_mem_size():
    """MAX-P: new_batch can exceed GSS mem_size; official max() over empty grads fails."""
    torch.manual_seed(0)
    plugin = AdaptiveGSSPlugin(mem_size=4, mem_strength=1, input_size=[4])
    model = Linear(4, 2)
    criterion = CrossEntropyLoss()

    def _step(n):
        strategy = _MbStrategy(
            model, criterion, torch.randn(n, 4), torch.zeros(n, dtype=torch.long)
        )
        plugin.device = torch.device("cpu")
        plugin.after_forward(strategy)

    _step(8)
    assert plugin.ext_mem_list_current_index == 4
    full_before = plugin.full_buffer_score_calls
    _step(8)
    assert plugin.ext_mem_list_current_index == 4
    assert plugin.full_buffer_score_calls == full_before + 1


def test_gss_replacement_all_equal_scores_does_not_crash():
    """First-insert scores are identical 0.1; official min-max then multinomial sums to 0."""
    torch.manual_seed(0)
    plugin = AdaptiveGSSPlugin(mem_size=4, mem_strength=1, input_size=[4])
    model = Linear(4, 2)
    plugin.ext_mem_list_x[:4] = torch.randn(4, 4)
    plugin.ext_mem_list_y[:4] = torch.zeros(4, dtype=torch.long)
    plugin.buffer_score[:4] = 0.1
    plugin.ext_mem_list_current_index = 4
    plugin.device = torch.device("cpu")
    n_params = sum(p.numel() for p in model.parameters())
    plugin.get_batch_sim = lambda *args, **kwargs: (
        torch.tensor(-0.5),
        torch.randn(1, n_params),
    )
    strategy = _MbStrategy(
        model, CrossEntropyLoss(), torch.randn(8, 4), torch.zeros(8, dtype=torch.long)
    )
    plugin.after_forward(strategy)
    assert plugin.ext_mem_list_current_index == 4


def test_gss_resize_does_not_restore_discarded():
    plugin = AdaptiveGSSPlugin(mem_size=6, mem_strength=1, input_size=[3])
    plugin.ext_mem_list_x[:4] = torch.arange(12, dtype=torch.float32).view(4, 3)
    plugin.ext_mem_list_y[:4] = torch.tensor([0, 1, 2, 3])
    plugin.buffer_score[:4] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    plugin.ext_mem_list_current_index = 4
    plugin.resize(2)
    assert plugin.mem_size == 2
    assert plugin.ext_mem_list_current_index == 2
    torch.testing.assert_close(plugin.ext_mem_list_y, torch.tensor([0, 1]))
    plugin.resize(6)
    assert plugin.mem_size == 6
    assert plugin.ext_mem_list_current_index == 2
    torch.testing.assert_close(plugin.ext_mem_list_y[:2], torch.tensor([0, 1]))


def test_gem_resize_does_not_restore_discarded():
    plugin = AdaptiveGEMPlugin(patterns_per_experience=4, memory_strength=0.5)
    plugin.memory_x[0] = torch.arange(16, dtype=torch.float32).view(4, 4)
    plugin.memory_y[0] = torch.tensor([0, 1, 2, 3])
    plugin.memory_tid[0] = torch.zeros(4, dtype=torch.long)
    plugin.resize(2)
    assert plugin.memory_x[0].shape[0] == 2
    torch.testing.assert_close(plugin.memory_y[0], torch.tensor([0, 1]))
    plugin.resize(8)
    assert plugin.patterns_per_experience == 8
    assert plugin.memory_x[0].shape[0] == 2


def test_toggle_off_on_skips_missing_gem_history():
    inner = AdaptiveGEMPlugin(patterns_per_experience=2, memory_strength=0.5)
    toggle = TogglePlugin(inner, enabled=False, name="gem")
    model = Linear(4, 2)
    strategy = SimpleNamespace(
        clock=SimpleNamespace(train_exp_counter=0),
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        _criterion=CrossEntropyLoss(),
        train_mb_size=2,
        experience=SimpleNamespace(dataset=[]),
    )
    toggle.after_training_exp(strategy)
    assert inner.memory_x == {}
    toggle.enabled = True
    strategy.clock.train_exp_counter = 1
    toggle.before_training_iteration(strategy)
    assert inner.G.numel() == 0
    toggle.enabled = False
    toggle.before_training_iteration(strategy)
    assert inner.G.numel() == 0
    toggle.enabled = True
    toggle.before_training_iteration(strategy)
    assert inner.G.numel() == 0


def test_apply_runtime_config_resizes_base_gem():
    spec = load_yaml(Path("configs/smoke_gem.yaml"))
    spec["training"]["device"] = "cpu"
    spec["prefetch"]["enabled"] = False
    model = build_model(spec)
    opt = build_optimizer(model, spec)
    strategy = build_strategy(model, opt, spec, device=torch.device("cpu"))
    gem = next(p for p in strategy.plugins if isinstance(p, AdaptiveGEMPlugin))
    gem.memory_x[0] = torch.zeros(50, 3, 32, 32)
    gem.memory_y[0] = torch.zeros(50, dtype=torch.long)
    gem.memory_tid[0] = torch.zeros(50, dtype=torch.long)
    n_exp = int(strategy._orion_n_experiences)
    state = apply_runtime_config(
        strategy, new_batch=8, replay_capacity=20, replay_batch=8
    )
    assert strategy.train_mb_size == 8
    assert gem.patterns_per_experience == patterns_per_experience(20, n_exp)
    assert gem.memory_x[0].shape[0] == gem.patterns_per_experience
    assert state["gem_patterns_per_experience"] == gem.patterns_per_experience
