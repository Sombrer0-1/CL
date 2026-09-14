"""Avalanche strategy construction. Semantics stay in the official plugins."""

from __future__ import annotations

from typing import Any, TextIO

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from orion_repro.models.resnet20 import resnet20
from orion_repro.prefetch.plugin import PrefetchWrapPlugin
from orion_repro.strategies.agem_compat import AdaptiveAGEMPlugin
from orion_repro.strategies.capacity import (
    agem_occupancy,
    gem_occupancy,
    gss_occupancy,
    iter_inner_plugins,
    patterns_per_experience,
    plugin_state_bytes,
)
from orion_repro.strategies.gem_compat import AdaptiveGEMPlugin, SparseGEMPlugin
from orion_repro.strategies.gss_compat import AdaptiveGSSPlugin, patch_gss_plugin
from orion_repro.strategies.latent_replay import LatentReplayPlugin
from orion_repro.strategies.toggles import TogglePlugin

ALGORITHMS_WITH_REPLAY_ADAPTER = {"er", "lr", "gem", "agem", "gss"}
DUPLICATE_GEM_BASES = {"gem"}
DUAL_PROJECTION_BASES = {"agem"}
GSS_GEM_MIX_BASES = {"gss"}


class UnsupportedAdaptationError(ValueError):
    """Dynamic Orion config applied to an algorithm with no adapter."""


def build_model(spec: dict[str, Any]) -> torch.nn.Module:
    name = spec["model"]["name"]
    num_classes = int(spec["model"]["num_classes"])
    if name != "cifar_resnet20":
        raise ValueError(f"unsupported model {name}")
    return resnet20(num_classes=num_classes)


def build_optimizer(model: torch.nn.Module, spec: dict[str, Any]) -> torch.optim.Optimizer:
    opt = spec["training"]["optimizer"]
    if opt["name"] != "sgd":
        raise ValueError(f"unsupported optimizer {opt['name']}")
    return SGD(
        model.parameters(),
        lr=float(opt["lr"]),
        momentum=float(opt.get("momentum", 0.0)),
        weight_decay=float(opt.get("weight_decay", 0.0)),
    )


def _optional_names(spec: dict[str, Any]) -> list[str]:
    raw = spec["algorithm"].get("optional_plugins", "none")
    if raw in (None, "none", "None", False):
        return []
    if raw == "gem_ewc":
        return ["gem", "ewc"]
    if raw in ("gem", "ewc"):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError(f"unsupported optional_plugins {raw!r}")


def validate_algorithm_combo(spec: dict[str, Any]) -> None:
    base = spec["algorithm"]["base"]
    extras = _optional_names(spec)
    if extras and base not in ALGORITHMS_WITH_REPLAY_ADAPTER:
        raise UnsupportedAdaptationError(
            f"optional_plugins on base={base} is not implemented"
        )
    if base in DUPLICATE_GEM_BASES and "gem" in extras:
        raise UnsupportedAdaptationError(
            "base=gem cannot stack another GEMPlugin; use optional_plugins=ewc "
            "for the A10 EWC-only reconstruction"
        )
    if base in DUAL_PROJECTION_BASES and "gem" in extras:
        raise UnsupportedAdaptationError(
            "base=agem + GEM is a dual gradient-projection mix; refusing silent "
            "combo. Use optional_plugins=ewc or none."
        )
    if base in GSS_GEM_MIX_BASES and "gem" in extras:
        raise UnsupportedAdaptationError(
            "base=gss + GEM mixes GSS selection with GEM projection; refusing "
            "silent combo. Use optional_plugins=ewc or none."
        )
    unknown = [name for name in extras if name not in {"gem", "ewc"}]
    if unknown:
        raise UnsupportedAdaptationError(f"unknown optional plugins {unknown}")


def _make_evaluator(log_file: TextIO | None):
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.logging import TextLogger
    from avalanche.training.plugins import EvaluationPlugin

    loggers = []
    if log_file is not None:
        loggers.append(TextLogger(log_file))
    return EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        loggers=loggers,
    )


def build_strategy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    *,
    device: torch.device,
    log_file: TextIO | None = None,
):
    from avalanche.training import Naive
    from avalanche.training.plugins import EWCPlugin, ReplayPlugin

    validate_algorithm_combo(spec)
    evaluator = _make_evaluator(log_file)
    algo = spec["algorithm"]
    base = algo["base"]
    mem_size = int(spec["replay"]["capacity"])
    new_batch = int(spec["training"]["new_batch"])
    replay_batch = int(spec["training"]["replay_batch"])
    train_epochs = int(spec["training"]["new_epochs"])
    eval_batch = int(spec["training"]["eval_batch"])
    version_holder = {"config_version": 0}
    prefetch_plugin = PrefetchWrapPlugin(
        enabled=bool(spec["prefetch"]["enabled"]),
        depth=int(spec["prefetch"].get("queue_depth", 1)),
        version_holder=version_holder,
        replay_seed=int(spec["seeds"]["replay"]),
    )
    common = dict(
        model=model,
        optimizer=optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size=new_batch,
        train_epochs=train_epochs,
        eval_mb_size=eval_batch,
        device=device,
        evaluator=evaluator,
        eval_every=-1,
    )
    extra: list = []
    n_experiences = int(
        spec["dataset"].get("n_experiences")
        or spec["dataset"].get("experience_limit")
        or 1
    )
    optional_enabled = bool(algo.get("optional_start_enabled", False))
    for name in _optional_names(spec):
        if name == "gem":
            extra.append(
                TogglePlugin(
                    SparseGEMPlugin(
                        patterns_per_experience=int(algo.get("patterns_per_exp", 50)),
                        memory_strength=float(algo.get("memory_strength", 0.5)),
                    ),
                    enabled=optional_enabled,
                    name="gem",
                )
            )
        elif name == "ewc":
            extra.append(
                TogglePlugin(
                    EWCPlugin(ewc_lambda=float(algo.get("ewc_lambda", 100.0))),
                    enabled=optional_enabled,
                    name="ewc",
                )
            )
        else:
            raise ValueError(f"unknown optional plugin {name}")
    if base == "er":
        extra.append(
            ReplayPlugin(mem_size=mem_size, batch_size=new_batch, batch_size_mem=replay_batch)
        )
    elif base == "lr":
        extra.append(LatentReplayPlugin(mem_size=mem_size, replay_batch=replay_batch))
    elif base == "gem":
        extra.append(
            AdaptiveGEMPlugin(
                patterns_per_experience=int(algo["patterns_per_exp"]),
                memory_strength=float(algo.get("memory_strength", 0.5)),
            )
        )
    elif base == "agem":
        extra.append(
            AdaptiveAGEMPlugin(
                patterns_per_experience=int(algo["patterns_per_exp"]),
                sample_size=int(algo.get("sample_size", replay_batch or 64)),
            )
        )
    elif base == "gss":
        patch_gss_plugin()
        extra.append(
            AdaptiveGSSPlugin(
                mem_size=mem_size,
                mem_strength=int(algo.get("mem_strength", 1)),
                input_size=list(algo["input_size"]),
            )
        )
    else:
        raise ValueError(f"algorithm {base} not wired yet")
    extra.append(prefetch_plugin)
    strategy = Naive(plugins=extra, **common)
    return _annotate(strategy, version_holder, prefetch_plugin, base, n_experiences)


def _annotate(
    strategy,
    version_holder: dict[str, int],
    prefetch_plugin: PrefetchWrapPlugin,
    base: str,
    n_experiences: int = 1,
):
    strategy._orion_version_holder = version_holder
    strategy._orion_prefetch_plugin = prefetch_plugin
    strategy._orion_base = base
    strategy._orion_n_experiences = int(n_experiences)
    return strategy


def plugin_audit(strategy) -> list[dict[str, Any]]:
    rows = []
    for plugin in getattr(strategy, "plugins", []):
        row = {
            "type": type(plugin).__name__,
            "name": getattr(plugin, "name", type(plugin).__name__),
        }
        if isinstance(plugin, TogglePlugin):
            row["enabled"] = bool(plugin.enabled)
            row["inner"] = type(plugin.inner).__name__
        rows.append(row)
    return rows


def _buffer_len(storage_policy) -> int:
    if storage_policy is None:
        return 0
    buf = getattr(storage_policy, "buffer", None)
    if buf is None:
        return 0
    try:
        return int(len(buf))
    except Exception:
        return 0


def replay_occupancy(strategy) -> dict[str, Any]:
    from avalanche.training.plugins import ReplayPlugin

    out: dict[str, Any] = {
        "replay_requested": None,
        "replay_max_size": None,
        "replay_occupancy": None,
        "latent_occupancy": None,
        "groups": None,
        "plugin_state_bytes": plugin_state_bytes(strategy),
    }
    for plugin in iter_inner_plugins(strategy):
        if isinstance(plugin, ReplayPlugin):
            policy = plugin.storage_policy
            out["replay_requested"] = int(plugin.mem_size)
            out["replay_max_size"] = int(getattr(policy, "max_size", plugin.mem_size))
            out["replay_occupancy"] = _buffer_len(policy)
            groups = getattr(policy, "buffer_groups", None)
            if isinstance(groups, dict):
                out["groups"] = {str(k): _buffer_len(v) for k, v in groups.items()}
        if isinstance(plugin, LatentReplayPlugin):
            n = 0 if plugin.buffer_z is None else int(plugin.buffer_z.shape[0])
            out["latent_occupancy"] = n
            out["replay_requested"] = int(plugin.mem_size)
            out["replay_occupancy"] = n
            out["replay_max_size"] = int(plugin.mem_size)
        if isinstance(plugin, AdaptiveGEMPlugin):
            out.update(gem_occupancy(plugin))
        elif isinstance(plugin, SparseGEMPlugin):
            extra = gem_occupancy(plugin)
            out["optional_gem_occupancy"] = extra["replay_occupancy"]
            out["optional_gem_patterns_per_experience"] = extra["gem_patterns_per_experience"]
        if isinstance(plugin, AdaptiveAGEMPlugin):
            out.update(agem_occupancy(plugin))
        if isinstance(plugin, AdaptiveGSSPlugin):
            out.update(gss_occupancy(plugin))
    out["plugin_state_bytes"] = plugin_state_bytes(strategy)
    return out


def apply_runtime_config(
    strategy,
    *,
    new_batch: int,
    replay_capacity: int,
    replay_batch: int,
    optimizer_mode: str | None = None,
) -> dict[str, Any]:
    from avalanche.training.plugins import ReplayPlugin

    base = getattr(strategy, "_orion_base", None)
    n_experiences = max(1, int(getattr(strategy, "_orion_n_experiences", 1)))
    strategy.train_mb_size = int(new_batch)
    adapted = False
    for plugin in list(strategy.plugins):
        inner = plugin.inner if isinstance(plugin, TogglePlugin) else plugin
        if isinstance(plugin, ReplayPlugin) or isinstance(inner, ReplayPlugin):
            target = plugin if isinstance(plugin, ReplayPlugin) else inner
            target.batch_size = new_batch
            target.batch_size_mem = replay_batch
            target.mem_size = int(replay_capacity)
            if target.storage_policy is not None:
                target.storage_policy.resize(strategy, int(replay_capacity))
            adapted = True
        if isinstance(inner, LatentReplayPlugin):
            inner.mem_size = int(replay_capacity)
            inner.replay_batch = replay_batch
            if inner.buffer_z is not None and inner.buffer_z.shape[0] > replay_capacity:
                inner.buffer_z = inner.buffer_z[:replay_capacity]
                inner.buffer_y = inner.buffer_y[:replay_capacity]
            adapted = True
        if isinstance(inner, AdaptiveGEMPlugin):
            inner.resize(patterns_per_experience(int(replay_capacity), n_experiences))
            adapted = True
        if isinstance(inner, AdaptiveAGEMPlugin):
            inner.resize(
                patterns_per_experience(int(replay_capacity), n_experiences),
                sample_size=max(1, int(replay_batch)) if replay_capacity > 0 else 0,
            )
            adapted = True
        if isinstance(inner, AdaptiveGSSPlugin):
            inner.resize(int(replay_capacity))
            adapted = True
        if isinstance(plugin, TogglePlugin) and optimizer_mode is not None:
            plugin.enabled = optimizer_mode == "advanced"
            adapted = True
    if not adapted:
        raise UnsupportedAdaptationError(
            f"apply_runtime_config found no adapter on base={base}"
        )
    holder = getattr(strategy, "_orion_version_holder", None)
    if isinstance(holder, dict):
        holder["config_version"] = int(holder.get("config_version", 0)) + 1
    state = replay_occupancy(strategy)
    state["applied_new_batch"] = int(new_batch)
    state["applied_replay_batch"] = int(replay_batch)
    state["applied_replay_capacity"] = int(replay_capacity)
    return state
