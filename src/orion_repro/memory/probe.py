"""Process/host/device resource snapshots (PLAN §8.1)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, TextIO

import psutil


@dataclass
class ResourceSnapshot:
    timestamp_s: float
    monotonic_s: float
    phase: str
    experience_index: int | None
    proc_rss_bytes: int
    children_rss_bytes: int
    proc_pss_bytes: int | None
    system_available_bytes: int | None
    swap_used_bytes: int | None
    gpu_alloc_bytes: int | None
    gpu_reserved_bytes: int | None
    gpu_alloc_peak_bytes: int | None
    gpu_global_used_bytes: int | None
    gpu_global_free_bytes: int | None
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _pss_or_none(proc: psutil.Process) -> int | None:
    try:
        mi = proc.memory_full_info()
        value = getattr(mi, "pss", None)
        return None if value is None else int(value)
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return None


def _gpu_fields() -> dict[str, int | None]:
    out = {
        "gpu_alloc_bytes": None,
        "gpu_reserved_bytes": None,
        "gpu_alloc_peak_bytes": None,
        "gpu_global_used_bytes": None,
        "gpu_global_free_bytes": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        out["gpu_alloc_bytes"] = int(torch.cuda.memory_allocated())
        out["gpu_reserved_bytes"] = int(torch.cuda.memory_reserved())
        out["gpu_alloc_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        try:
            free, total = torch.cuda.mem_get_info()
            out["gpu_global_free_bytes"] = int(free)
            out["gpu_global_used_bytes"] = int(total - free)
        except Exception:
            pass
    except Exception:
        pass
    return out


def snapshot(phase: str, experience_index: int | None = None, notes: str = "") -> ResourceSnapshot:
    proc = psutil.Process()
    rss = int(proc.memory_info().rss)
    child_rss = 0
    for ch in proc.children(recursive=True):
        try:
            child_rss += int(ch.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu = _gpu_fields()
    return ResourceSnapshot(
        timestamp_s=time.time(),
        monotonic_s=time.monotonic(),
        phase=phase,
        experience_index=experience_index,
        proc_rss_bytes=rss,
        children_rss_bytes=child_rss,
        proc_pss_bytes=_pss_or_none(proc),
        system_available_bytes=int(vm.available),
        swap_used_bytes=int(swap.used),
        notes=notes,
        **gpu,
    )


class ResourceSampler:
    """Background RSS/allocator sampler. Peak is an observed lower bound."""

    def __init__(self, interval_s: float, writer: TextIO | None = None) -> None:
        self.interval_s = interval_s
        self.writer = writer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "idle"
        self._exp: int | None = None
        self.peak_rss = 0
        self.peak_gpu_alloc = 0
        self.phase_peak_rss = 0
        self.phase_peak_gpu_alloc = 0
        self.rows: list[ResourceSnapshot] = []

    def set_phase(self, phase: str, experience_index: int | None) -> None:
        self._phase = phase
        self._exp = experience_index
        self.phase_peak_rss = 0
        self.phase_peak_gpu_alloc = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 3 * self.interval_s))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = snapshot(self._phase, self._exp, notes="interval")
            self.rows.append(snap)
            rss = snap.proc_rss_bytes + snap.children_rss_bytes
            self.peak_rss = max(self.peak_rss, rss)
            self.phase_peak_rss = max(self.phase_peak_rss, rss)
            gpu = snap.gpu_alloc_peak_bytes or snap.gpu_alloc_bytes or 0
            if gpu:
                self.peak_gpu_alloc = max(self.peak_gpu_alloc, int(gpu))
                self.phase_peak_gpu_alloc = max(self.phase_peak_gpu_alloc, int(gpu))
            if self.writer is not None:
                import csv

                row = snap.as_row()
                writer = csv.DictWriter(self.writer, fieldnames=list(row.keys()))
                if self.writer.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
                self.writer.flush()
            self._stop.wait(self.interval_s)


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def synchronize_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def cgroup_memory_current() -> int | None:
    candidates = [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return int(PathRead(path))
            except Exception:
                return None
    return None


def PathRead(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.read().strip())
