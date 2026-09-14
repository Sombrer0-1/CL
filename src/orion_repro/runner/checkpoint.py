"""Experience-boundary training-state checkpoint (PLAN §8.6 / T10).

This is not matrix --resume, which only reuses completed runs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orion_repro.strategies.agem_compat import AdaptiveAGEMPlugin
from orion_repro.strategies.gem_compat import AdaptiveGEMPlugin, SparseGEMPlugin
from orion_repro.strategies.gss_compat import AdaptiveGSSPlugin
from orion_repro.strategies.latent_replay import LatentReplayPlugin
from orion_repro.strategies.toggles import TogglePlugin


def capture_rng() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def serialize_classification_dataset(dataset) -> dict[str, Any]:
    """Store (x, y[, task]) tensors from an Avalanche/iterable classification set."""
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    ts: list[int] = []
    has_task = False
    if dataset is None:
        return {"n": 0, "x": torch.zeros(0), "y": torch.zeros(0, dtype=torch.long)}
    for sample in dataset:
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError(f"unsupported dataset sample type {type(sample)}")
        xs.append(_as_cpu_tensor(sample[0]))
        y = sample[1]
        ys.append(int(y.item()) if torch.is_tensor(y) else int(y))
        if len(sample) > 2:
            has_task = True
            t = sample[2]
            ts.append(int(t.item()) if torch.is_tensor(t) else int(t))
    if not xs:
        return {"n": 0, "x": torch.zeros(0), "y": torch.zeros(0, dtype=torch.long)}
    payload: dict[str, Any] = {
        "n": len(xs),
        "x": torch.stack(xs, dim=0),
        "y": torch.tensor(ys, dtype=torch.long),
    }
    if has_task:
        payload["t"] = torch.tensor(ts, dtype=torch.long)
    return payload


class _LabeledTensorDataset:
    """Minimal dataset with a `targets` field for Avalanche ClassificationDataset."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None) -> None:
        self.x = x
        self.y = y
        self.t = t
        self.targets = [int(v) for v in y.tolist()]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        if self.t is None:
            return self.x[index], int(self.y[index])
        return self.x[index], int(self.y[index]), int(self.t[index])


def dataset_from_tensors(payload: dict[str, Any]):
    from avalanche.benchmarks.utils.utils import as_classification_dataset

    n = int(payload.get("n", 0))
    if n <= 0:
        from avalanche.benchmarks.utils.utils import concat_datasets

        return concat_datasets([])
    return as_classification_dataset(
        _LabeledTensorDataset(payload["x"], payload["y"], payload.get("t"))
    )


def serialize_plugins(strategy) -> list[dict[str, Any]]:
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.training.storage_policy import ExperienceBalancedBuffer

    rows: list[dict[str, Any]] = []
    for plugin in getattr(strategy, "plugins", []):
        inner = plugin.inner if isinstance(plugin, TogglePlugin) else plugin
        row: dict[str, Any] = {
            "type": type(inner).__name__,
            "toggle_enabled": bool(plugin.enabled) if isinstance(plugin, TogglePlugin) else None,
        }
        if isinstance(inner, (AdaptiveGEMPlugin, SparseGEMPlugin)):
            row["patterns_per_experience"] = int(inner.patterns_per_experience)
            row["memory_x"] = {str(k): v.detach().cpu() for k, v in inner.memory_x.items()}
            row["memory_y"] = {str(k): v.detach().cpu() for k, v in inner.memory_y.items()}
            row["memory_tid"] = {str(k): v.detach().cpu() for k, v in inner.memory_tid.items()}
        if isinstance(inner, AdaptiveGSSPlugin):
            row["mem_size"] = int(inner.mem_size)
            row["current_index"] = int(inner.ext_mem_list_current_index)
            row["ext_mem_list_x"] = inner.ext_mem_list_x.detach().cpu()
            row["ext_mem_list_y"] = inner.ext_mem_list_y.detach().cpu()
            row["buffer_score"] = inner.buffer_score.detach().cpu()
        if isinstance(inner, LatentReplayPlugin):
            row["mem_size"] = int(inner.mem_size)
            row["frozen"] = bool(inner._frozen)
            row["buffer_z"] = None if inner.buffer_z is None else inner.buffer_z.detach().cpu()
            row["buffer_y"] = None if inner.buffer_y is None else inner.buffer_y.detach().cpu()
        if isinstance(inner, AdaptiveAGEMPlugin):
            row["patterns_per_experience"] = int(inner.patterns_per_experience)
            row["sample_size"] = int(inner.sample_size)
            row["agem_n_buffers"] = len(inner.buffers)
            row["agem_buffers"] = [serialize_classification_dataset(buf) for buf in inner.buffers]
            row["agem_buffer_note"] = "serialized as CPU tensors; dataloader rebuilt on restore"
        if isinstance(inner, ReplayPlugin):
            policy = inner.storage_policy
            row["mem_size"] = int(inner.mem_size)
            row["max_size"] = int(getattr(policy, "max_size", inner.mem_size))
            buf = getattr(policy, "buffer", None)
            row["replay_occupancy"] = 0 if buf is None else int(len(buf))
            if isinstance(policy, ExperienceBalancedBuffer):
                row["policy_max_size"] = int(policy.max_size)
                row["_num_exps"] = int(getattr(policy, "_num_exps", len(policy.buffer_groups)))
                groups = {}
                for key, group in policy.buffer_groups.items():
                    groups[str(key)] = {
                        "max_size": int(group.max_size),
                        "weights": getattr(group, "_buffer_weights", torch.zeros(0)).detach().cpu(),
                        **serialize_classification_dataset(group.buffer),
                    }
                row["buffer_groups"] = groups
        rows.append(row)
    return rows


def restore_plugins(strategy, rows: list[dict[str, Any]]) -> None:
    plugins = list(getattr(strategy, "plugins", []))
    if len(plugins) != len(rows):
        raise ValueError(f"checkpoint plugin count {len(rows)} != strategy {len(plugins)}")
    for plugin, row in zip(plugins, rows):
        inner = plugin.inner if isinstance(plugin, TogglePlugin) else plugin
        if isinstance(plugin, TogglePlugin) and row.get("toggle_enabled") is not None:
            plugin.enabled = bool(row["toggle_enabled"])
        if isinstance(inner, (AdaptiveGEMPlugin, SparseGEMPlugin)):
            inner.patterns_per_experience = int(row["patterns_per_experience"])
            inner.memory_x = {int(k): v.clone() for k, v in row["memory_x"].items()}
            inner.memory_y = {int(k): v.clone() for k, v in row["memory_y"].items()}
            inner.memory_tid = {int(k): v.clone() for k, v in row["memory_tid"].items()}
        if isinstance(inner, AdaptiveGSSPlugin):
            inner.resize(int(row["mem_size"]))
            inner.ext_mem_list_x.copy_(row["ext_mem_list_x"].to(inner.ext_mem_list_x.device))
            inner.ext_mem_list_y.copy_(row["ext_mem_list_y"].to(inner.ext_mem_list_y.device))
            inner.buffer_score.copy_(row["buffer_score"].to(inner.buffer_score.device))
            inner.ext_mem_list_current_index = int(row["current_index"])
        if isinstance(inner, LatentReplayPlugin):
            inner.mem_size = int(row["mem_size"])
            inner._frozen = bool(row["frozen"])
            inner.buffer_z = None if row["buffer_z"] is None else row["buffer_z"].clone()
            inner.buffer_y = None if row["buffer_y"] is None else row["buffer_y"].clone()
        if isinstance(inner, AdaptiveAGEMPlugin) and "agem_buffers" in row:
            inner.patterns_per_experience = int(row["patterns_per_experience"])
            inner.sample_size = int(row["sample_size"])
            inner.buffers = [dataset_from_tensors(item) for item in row["agem_buffers"]]
            inner.resize(inner.patterns_per_experience, sample_size=inner.sample_size)
        from avalanche.training.plugins import ReplayPlugin
        from avalanche.training.storage_policy import ExperienceBalancedBuffer, ReservoirSamplingBuffer

        if isinstance(inner, ReplayPlugin) and "buffer_groups" in row:
            inner.mem_size = int(row["mem_size"])
            policy = inner.storage_policy
            if isinstance(policy, ExperienceBalancedBuffer):
                policy.max_size = int(row.get("policy_max_size", row["max_size"]))
                policy._num_exps = int(row.get("_num_exps", len(row["buffer_groups"])))
                policy.buffer_groups.clear()
                for key, group in row["buffer_groups"].items():
                    buf = ReservoirSamplingBuffer(int(group["max_size"]))
                    buf.buffer = dataset_from_tensors(group)
                    weights = group.get("weights")
                    if weights is not None:
                        buf._buffer_weights = weights.clone()
                    policy.buffer_groups[int(key)] = buf


def save_checkpoint(
    path: Path,
    *,
    strategy,
    ctrl_state,
    completed_experiences: int,
    matrix: np.ndarray,
    correct_mat: np.ndarray,
    totals: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_experiences": int(completed_experiences),
        "model": {k: v.detach().cpu() for k, v in strategy.model.state_dict().items()},
        "optimizer": strategy.optimizer.state_dict(),
        "train_mb_size": int(strategy.train_mb_size),
        "rng": capture_rng(),
        "plugins": serialize_plugins(strategy),
        "controller": None
        if ctrl_state is None
        else {
            "mb": float(ctrl_state.mb),
            "mr": float(ctrl_state.mr),
            "new_batch": int(ctrl_state.new_batch),
            "replay_capacity": int(ctrl_state.replay_capacity),
            "optimizer_mode": str(ctrl_state.optimizer_mode),
        },
        "matrix": np.asarray(matrix),
        "correct_mat": np.asarray(correct_mat),
        "totals": np.asarray(totals),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_checkpoint(strategy, payload: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    strategy.model.load_state_dict(payload["model"])
    strategy.model.to(device)
    strategy.optimizer.load_state_dict(payload["optimizer"])
    strategy.train_mb_size = int(payload["train_mb_size"])
    restore_plugins(strategy, payload["plugins"])
    restore_rng(payload["rng"])
    return payload
