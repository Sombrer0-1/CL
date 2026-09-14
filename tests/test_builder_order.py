from pathlib import Path

import torch

from orion_repro.runner.spec import load_yaml
from orion_repro.strategies.builder import build_model, build_optimizer, build_strategy, plugin_audit
from orion_repro.strategies.latent_replay import LatentReplayPlugin
from orion_repro.prefetch.plugin import PrefetchWrapPlugin
from orion_repro.strategies.toggles import TogglePlugin


def _strategy(config: str, *, prefetch: bool | None = None):
    spec = load_yaml(Path(config))
    if prefetch is not None:
        spec["prefetch"]["enabled"] = prefetch
    model = build_model(spec)
    opt = build_optimizer(model, spec)
    return build_strategy(model, opt, spec, device=torch.device("cpu"))


def test_er_plugin_order_prefetch_after_replay():
    from avalanche.training.plugins import ReplayPlugin

    strategy = _strategy("configs/smoke_prefetch_on.yaml")
    types = [type(p).__name__ for p in strategy.plugins]
    assert "ReplayPlugin" in types
    assert types.index("ReplayPlugin") < types.index("PrefetchWrapPlugin")
    assert any(isinstance(p, PrefetchWrapPlugin) for p in strategy.plugins)
    assert any(isinstance(p, ReplayPlugin) for p in strategy.plugins)


def test_max_a_toggles_start_enabled_before_replay():
    strategy = _strategy("configs/smoke_max_a.yaml")
    names = plugin_audit(strategy)
    kinds = [row["type"] for row in names]
    assert kinds.index("TogglePlugin") < kinds.index("ReplayPlugin")
    assert kinds.index("ReplayPlugin") < kinds.index("PrefetchWrapPlugin")
    toggles = [p for p in strategy.plugins if isinstance(p, TogglePlugin)]
    assert {p.name for p in toggles} == {"gem", "ewc"}
    assert all(p.enabled for p in toggles)


def test_base_gem_uses_adaptive_plugin_before_prefetch():
    from orion_repro.strategies.gem_compat import AdaptiveGEMPlugin

    strategy = _strategy("configs/smoke_gem.yaml")
    types = [type(p).__name__ for p in strategy.plugins]
    assert types.index("AdaptiveGEMPlugin") < types.index("PrefetchWrapPlugin")
    assert any(isinstance(p, AdaptiveGEMPlugin) for p in strategy.plugins)


def test_lr_uses_latent_plugin():
    strategy = _strategy("configs/smoke_lr.yaml")
    assert any(isinstance(p, LatentReplayPlugin) for p in strategy.plugins)
