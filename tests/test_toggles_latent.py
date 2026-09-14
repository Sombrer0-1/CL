import torch

from orion_repro.models.resnet20 import resnet20
from orion_repro.strategies.toggles import TogglePlugin


class _Probe:
    def __init__(self):
        self.calls = []

    def before_training_exp(self, strategy, **kwargs):
        self.calls.append("before_training_exp")

    def after_backward(self, strategy, **kwargs):
        self.calls.append("after_backward")


def test_toggle_skips_hooks_when_disabled():
    inner = _Probe()
    plugin = TogglePlugin(inner, enabled=False, name="probe")
    plugin.before_training_exp(None)
    plugin.after_backward(None)
    assert inner.calls == []
    plugin.enabled = True
    plugin.before_training_exp(None)
    assert inner.calls == ["before_training_exp"]


def test_sparse_gem_skips_missing_memories():
    from types import SimpleNamespace

    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD

    from orion_repro.models.resnet20 import resnet20
    from orion_repro.strategies.gem_compat import SparseGEMPlugin

    net = resnet20(num_classes=10)
    plugin = SparseGEMPlugin(patterns_per_experience=4, memory_strength=0.5)
    strategy = SimpleNamespace()
    strategy.clock = SimpleNamespace(train_exp_counter=1)
    strategy.model = net
    strategy.optimizer = SGD(net.parameters(), lr=0.01)
    strategy.device = torch.device("cpu")
    strategy._criterion = CrossEntropyLoss()
    plugin.before_training_iteration(strategy)
    assert plugin.G.numel() == 0
    plugin.after_backward(strategy)


def test_sparse_gem_qp_none_keeps_gradient(monkeypatch):
    from orion_repro.strategies.gem_compat import SparseGEMPlugin
    import qpsolvers

    plugin = SparseGEMPlugin(patterns_per_experience=4, memory_strength=0.5)
    plugin.G = torch.eye(2)
    plugin.memory_strength = 0.5
    monkeypatch.setattr(qpsolvers, "solve_qp", lambda **kwargs: None)
    g = torch.arange(2).float()
    out = plugin.solve_quadprog(g)
    assert plugin.qp_failures == 1
    torch.testing.assert_close(out, g)


def test_latent_freeze_concatenates_replay():
    model = resnet20(num_classes=10)
    model._latent_freeze_lower = True
    for module in (model.conv1, model.bn1, model.layer1, model.layer2):
        for param in module.parameters():
            param.requires_grad = False
    model.train()
    assert model.bn1.training is False
    replay = torch.randn(2, 32, 16, 16)
    model._latent_replay_z = replay
    out = model(torch.randn(3, 3, 32, 32))
    assert out.shape == (5, 10)
