"""Wrap an already-built DataLoader with a bounded CPU producer (PLAN §7.5)."""

from __future__ import annotations

from typing import Any, Callable, Iterator

from orion_repro.prefetch.queue import BoundedPrefetcher, PrefetchItem


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
    ) -> None:
        self.loader = loader
        self.experience_id = int(experience_id)
        self.config_version = int(config_version)
        self.max_depth = int(max_depth)
        self.use_queue = bool(use_queue) and self.max_depth > 0
        self.prefetcher = BoundedPrefetcher(self.max_depth) if self.use_queue else None
        self.wait_s = 0.0
        self.batches_consumed = 0
        self.planned_index_batches: list[list[int]] | None = None
        self.plan_mode = "unplanned_fallback"

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        planned = plan_index_batches(self.loader)
        if planned is not None:
            index_batches, dataset, collate = planned
            self.planned_index_batches = index_batches
            if not self.use_queue:
                self.plan_mode = "main_thread_indices_serial"
                for indices in index_batches:
                    samples = [dataset[i] for i in indices]
                    self.batches_consumed += 1
                    yield collate(samples)
                return
            self.plan_mode = "main_thread_indices"

            def producer():
                for step, indices in enumerate(index_batches):
                    samples = [dataset[i] for i in indices]
                    batch = collate(samples)
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
                for batch in inner_iter:
                    self.batches_consumed += 1
                    yield batch
                return
            self.plan_mode = "unplanned_fallback"

            def producer():
                for step, batch in enumerate(inner_iter):
                    yield PrefetchItem(
                        experience_id=self.experience_id,
                        config_version=self.config_version,
                        step_id=step,
                        batch=batch,
                    )

        assert self.prefetcher is not None
        self.prefetcher.start(producer)
        try:
            while True:
                item = self.prefetcher.get(config_version=self.config_version)
                if item is None:
                    break
                self.batches_consumed += 1
                yield item.batch
        finally:
            self.wait_s = self.prefetcher.wait_s
            self.prefetcher.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)
