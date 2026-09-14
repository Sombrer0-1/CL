"""Resize GEM/AGEM/GSS memories without restoring discarded samples.

URGE MR is a total sample-slot budget. GEM/AGEM store patterns per experience;
GSS stores a single buffer. Mapping is recorded in A10, not a paper formula.
"""

from __future__ import annotations

from typing import Any


def patterns_per_experience(capacity: int, n_experiences: int) -> int:
    """Map total replay slots onto a per-experience GEM/AGEM cap.

    Truncation is toward zero; capacity 0 yields 0 patterns (no constraint).
    """
    if n_experiences <= 0:
        raise ValueError("n_experiences must be positive")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    return int(capacity) // int(n_experiences)


def _unwrap(plugin: Any) -> Any:
    inner = getattr(plugin, "inner", None)
    return inner if inner is not None else plugin


def iter_inner_plugins(strategy) -> list[Any]:
    return [_unwrap(p) for p in getattr(strategy, "plugins", [])]


def gem_occupancy(plugin) -> dict[str, Any]:
    stored = 0
    per_exp: dict[str, int] = {}
    memory_x = getattr(plugin, "memory_x", {}) or {}
    for key, tensor in memory_x.items():
        n = 0 if tensor is None else int(tensor.shape[0])
        per_exp[str(key)] = n
        stored += n
    ppe = int(getattr(plugin, "patterns_per_experience", 0))
    return {
        "replay_occupancy": stored,
        "replay_requested": ppe * max(len(memory_x), 1) if memory_x else ppe,
        "replay_max_size": ppe * max(len(memory_x), 1) if memory_x else ppe,
        "gem_patterns_per_experience": ppe,
        "gem_experiences_stored": len(memory_x),
        "groups": per_exp,
    }


def agem_occupancy(plugin) -> dict[str, Any]:
    buffers = list(getattr(plugin, "buffers", []) or [])
    stored = sum(int(len(buf)) for buf in buffers)
    ppe = int(getattr(plugin, "patterns_per_experience", 0))
    return {
        "replay_occupancy": stored,
        "replay_requested": ppe * max(len(buffers), 1) if buffers else ppe,
        "replay_max_size": ppe * max(len(buffers), 1) if buffers else ppe,
        "agem_patterns_per_experience": ppe,
        "agem_sample_size": int(getattr(plugin, "sample_size", 0)),
        "agem_experiences_stored": len(buffers),
    }


def gss_occupancy(plugin) -> dict[str, Any]:
    current = int(getattr(plugin, "ext_mem_list_current_index", 0))
    mem_size = int(getattr(plugin, "mem_size", 0))
    return {
        "replay_occupancy": current,
        "replay_requested": mem_size,
        "replay_max_size": mem_size,
        "gss_mem_size": mem_size,
    }


def collect_auxiliary_visits(strategy) -> dict[str, Any]:
    total = 0
    scopes: list[str] = []
    measured = False
    for plugin in iter_inner_plugins(strategy):
        if hasattr(plugin, "auxiliary_visits"):
            measured = True
            total += int(getattr(plugin, "auxiliary_visits") or 0)
            scopes.append(type(plugin).__name__)
    if not measured:
        return {
            "auxiliary_visits": None,
            "auxiliary_visit_scope": "unmeasured_algorithm_specific",
        }
    return {
        "auxiliary_visits": total,
        "auxiliary_visit_scope": ",".join(scopes) if scopes else "none",
    }


def reset_auxiliary_visits(strategy) -> None:
    for plugin in iter_inner_plugins(strategy):
        if hasattr(plugin, "auxiliary_visits"):
            plugin.auxiliary_visits = 0


def plugin_state_bytes(strategy) -> dict[str, int]:
    """Tensor nbytes currently held by algorithm plugins. Host or device as stored."""
    out: dict[str, int] = {}
    for plugin in iter_inner_plugins(strategy):
        name = type(plugin).__name__
        nbytes = 0
        memory_x = getattr(plugin, "memory_x", None)
        if isinstance(memory_x, dict):
            for tensor in memory_x.values():
                nbytes += int(tensor.nbytes)
            for attr in ("memory_y", "memory_tid"):
                mapping = getattr(plugin, attr, {}) or {}
                for tensor in mapping.values():
                    nbytes += int(tensor.nbytes)
        for attr in ("ext_mem_list_x", "ext_mem_list_y", "buffer_score", "buffer_z", "buffer_y"):
            tensor = getattr(plugin, attr, None)
            if tensor is not None and hasattr(tensor, "nbytes"):
                nbytes += int(tensor.nbytes)
        buffers = getattr(plugin, "buffers", None)
        if buffers:
            for buf in buffers:
                try:
                    nbytes += int(len(buf)) * 4
                except TypeError:
                    pass
        saved = getattr(plugin, "saved_params", None)
        importances = getattr(plugin, "importances", None)
        for store in (saved, importances):
            if not store:
                continue
            for exp_map in store.values():
                if not isinstance(exp_map, dict):
                    continue
                for value in exp_map.values():
                    data = getattr(value, "data", value)
                    if hasattr(data, "nbytes"):
                        nbytes += int(data.nbytes)
        out[name] = out.get(name, 0) + nbytes
    return out
