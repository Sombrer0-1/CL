"""Avalanche plugin that wraps strategy.dataloader after ReplayPlugin."""

from __future__ import annotations

from typing import Any

from avalanche.core import SupervisedPlugin

from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.rng import mix_seed, torch_generator


class PrefetchWrapPlugin(SupervisedPlugin):
    def __init__(
        self,
        enabled: bool,
        depth: int,
        version_holder: dict[str, int],
        replay_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.depth = max(1, int(depth))
        self.version_holder = version_holder
        self.replay_seed = replay_seed
        self.last_wait_s = 0.0
        self.last_dropped_stale = 0
        self.last_batches = 0
        self.last_plan_mode = "unused"
        self.last_planned_n = 0
        self.last_visits = {}
        self._wrapped: PrefetchingDataLoader | None = None

    def before_training_exp(self, strategy, *args, **kwargs) -> None:
        inner = strategy.dataloader
        self._new_length = len(strategy.adapted_dataset)
        self._is_replay = type(inner).__name__ == "ReplayDataLoader"
        exp_id = int(getattr(getattr(strategy, "clock", None), "train_exp_counter", 0) or 0)
        if self.replay_seed is not None and hasattr(inner, "loader_kwargs"):
            inner.loader_kwargs["generator"] = torch_generator(
                mix_seed(int(self.replay_seed), exp_id, "replay_loader")
            )
        self._wrapped = PrefetchingDataLoader(
            inner,
            experience_id=exp_id,
            config_version=int(self.version_holder.get("config_version", 0)),
            max_depth=self.depth,
            use_queue=self.enabled,
        )
        strategy.dataloader = self._wrapped

    def after_training_exp(self, strategy, *args, **kwargs) -> None:
        if self._wrapped is not None:
            self.last_wait_s = float(self._wrapped.wait_s)
            dropped = 0
            if self._wrapped.prefetcher is not None:
                dropped = int(self._wrapped.prefetcher.dropped_stale)
            self.last_dropped_stale = dropped
            self.last_batches = int(self._wrapped.batches_consumed)
            self.last_plan_mode = str(self._wrapped.plan_mode)
            planned = self._wrapped.planned_index_batches
            self.last_planned_n = 0 if planned is None else sum(len(b) for b in planned)
            if planned is not None:
                from collections import Counter
                counts = Counter(i for batch in planned[:self.last_batches] for i in batch)
                new = {i: n for i, n in counts.items() if i < self._new_length}
                self.last_visits = {
                    "new_sgd_visits": sum(new.values()),
                    "new_unique_samples": len(new),
                    "new_missing_samples": self._new_length - len(new),
                    "new_repeated_visits": sum(max(0, n - 1) for n in new.values()),
                    "replay_visits": sum(n for i, n in counts.items() if i >= self._new_length) if self._is_replay else 0,
                    "visit_scope": "consumed_main_loader_indices",
                }
            else:
                self.last_visits = {"new_sgd_visits": None, "replay_visits": None,
                    "visit_scope": "unavailable_unplanned_loader"}
            self._wrapped = None
