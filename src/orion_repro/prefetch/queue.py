"""Bounded prefetch queue (PLAN §7.5). Producer only transforms planned batches."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PrefetchItem:
    experience_id: int
    config_version: int
    step_id: int
    batch: Any


class PrefetchError(RuntimeError):
    pass


class BoundedPrefetcher:
    def __init__(self, max_depth: int) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self._q: queue.Queue = queue.Queue(maxsize=max_depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.dropped_stale = 0
        self.enqueued = 0
        self.wait_s = 0.0

    def start(self, producer: Callable[[], Iterable[PrefetchItem]]) -> None:
        self._stop.clear()
        self._error = None
        self.wait_s = 0.0

        def _enqueue(item: PrefetchItem | None) -> None:
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            if item is None:
                try:
                    self._q.put_nowait(None)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(None)
                    except queue.Full:
                        pass

        def _run() -> None:
            try:
                for item in producer():
                    if self._stop.is_set():
                        break
                    _enqueue(item)
                    if not self._stop.is_set():
                        self.enqueued += 1
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                _enqueue(None)

        self._thread = threading.Thread(target=_run, name="orion-prefetcher", daemon=True)
        self._thread.start()

    def get(self, *, config_version: int, timeout: float | None = None) -> PrefetchItem | None:
        import time

        while True:
            t0 = time.monotonic()
            item = self._q.get(timeout=timeout)
            self.wait_s += time.monotonic() - t0
            if item is None:
                if self._error is not None:
                    raise PrefetchError("producer failed") from self._error
                return None
            if item.config_version != config_version:
                self.dropped_stale += 1
                continue
            return item

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
