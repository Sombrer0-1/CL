"""Wrap an already-built DataLoader with a bounded CPU producer (PLAN §7.5)."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Iterator

from orion_repro.prefetch.queue import BoundedPrefetcher, PrefetchItem
from orion_repro.prefetch.supply import extract_xy, tensor_sha256, update_rolling


def _identity_collate(batch):
    return batch[0] if len(batch) == 1 else batch


def plan_index_batches(loader: Any) -> tuple[list[list[int]], Any, Callable] | None:
    """Materialize batch index lists on the caller thread.

    Returns (index_batches, dataset, collate_fn) or None if the loader has no
    inspectable batch sampler. Producer then only reads those indices.
    """
    torch_loader = None
    get_loader = getattr(loader, "_get_loader", None)
    if callable(get_loader):
        torch_loader = get_loader()
    elif hasattr(loader, "batch_sampler") and getattr(loader, "dataset", None) is not None:
        torch_loader = loader
    if torch_loader is None:
        return None
    batch_sampler = getattr(torch_loader, "batch_sampler", None)
    dataset = getattr(torch_loader, "dataset", None)
    if batch_sampler is None or dataset is None:
        return None
    planned = [list(batch) for batch in batch_sampler]
    collate = getattr(torch_loader, "collate_fn", None) or _identity_collate
    return planned, dataset, collate


class PrefetchingDataLoader:
    """Bounded prefetch. Sample IDs are planned on the main thread when possible."""

    def __init__(
        self,
        loader: Any,
        *,
        experience_id: int,
        config_version: int,
        max_depth: int,
        use_queue: bool = True,
        record_hashes: bool = False,
    ) -> None:
        self.loader = loader
        self.experience_id = int(experience_id)
        self.config_version = int(config_version)
        self.max_depth = int(max_depth)
        self.use_queue = bool(use_queue) and self.max_depth > 0
        self.prefetcher = BoundedPrefetcher(self.max_depth) if self.use_queue else None
        self.wait_s = 0.0
        self.produce_s = 0.0
        self.batches_consumed = 0
        self.planned_index_batches: list[list[int]] | None = None
        self.plan_mode = "unplanned_fallback"
        self.record_hashes = bool(record_hashes)
        self.rolling = hashlib.sha256()
        self.first_x_hash: str | None = None
        self.first_y_hash: str | None = None
        self.last_x_hash: str | None = None
        self.last_y_hash: str | None = None

    def _observe(self, batch: Any) -> None:
        if not self.record_hashes:
            return
        x, y = extract_xy(batch)
        hx = tensor_sha256(x)
        hy = tensor_sha256(y)
        if self.first_x_hash is None:
            self.first_x_hash, self.first_y_hash = hx, hy
        self.last_x_hash, self.last_y_hash = hx, hy
        update_rolling(self.rolling, hx, hy)

    def digest(self) -> dict[str, Any]:
        return {
            "batches_consumed": int(self.batches_consumed),
            "produce_s": float(self.produce_s),
            "wait_s": float(self.wait_s),
            "rolling_sha256": self.rolling.hexdigest() if self.record_hashes else None,
            "first_x_hash": self.first_x_hash,
            "first_y_hash": self.first_y_hash,
            "last_x_hash": self.last_x_hash,
            "last_y_hash": self.last_y_hash,
            "plan_mode": self.plan_mode,
        }

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        self.wait_s = 0.0
        self.produce_s = 0.0
        self.batches_consumed = 0
        self.planned_index_batches = None
        self.rolling = hashlib.sha256()
        self.first_x_hash = None
        self.first_y_hash = None
        self.last_x_hash = None
        self.last_y_hash = None
        planned = plan_index_batches(self.loader)
        if planned is not None:
            index_batches, dataset, collate = planned
            self.planned_index_batches = index_batches
            if not self.use_queue:
                self.plan_mode = "main_thread_indices_serial"
                for indices in index_batches:
                    t0 = time.perf_counter()
                    samples = [dataset[i] for i in indices]
                    batch = collate(samples)
                    self.produce_s += time.perf_counter() - t0
                    self.batches_consumed += 1
                    self._observe(batch)
                    yield batch
                return
            self.plan_mode = "main_thread_indices"

            def producer():
                for step, indices in enumerate(index_batches):
                    t0 = time.perf_counter()
                    samples = [dataset[i] for i in indices]
                    batch = collate(samples)
                    self.produce_s += time.perf_counter() - t0
                    yield PrefetchItem(
                        experience_id=self.experience_id,
                        config_version=self.config_version,
                        step_id=step,
                        batch=batch,
                    )

        else:
            inner_iter = iter(self.loader)
            if not self.use_queue:
                self.plan_mode = "unplanned_serial"
                while True:
                    t0 = time.perf_counter()
                    try:
                        batch = next(inner_iter)
                    except StopIteration:
                        break
                    self.produce_s += time.perf_counter() - t0
                    self.batches_consumed += 1
                    self._observe(batch)
                    yield batch
                return
            self.plan_mode = "unplanned_fallback"

            def producer():
                step = 0
                while True:
                    t0 = time.perf_counter()
                    try:
                        batch = next(inner_iter)
                    except StopIteration:
                        break
                    self.produce_s += time.perf_counter() - t0
                    yield PrefetchItem(
                        experience_id=self.experience_id,
                        config_version=self.config_version,
                        step_id=step,
                        batch=batch,
                    )
                    step += 1

        assert self.prefetcher is not None
        self.prefetcher.start(producer)
        try:
            while True:
                item = self.prefetcher.get(
                    config_version=self.config_version,
                    experience_id=self.experience_id,
                )
                if item is None:
                    break
                self.batches_consumed += 1
                self._observe(item.batch)
                yield item.batch
        finally:
            self.wait_s = self.prefetcher.wait_s
            self.prefetcher.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)
