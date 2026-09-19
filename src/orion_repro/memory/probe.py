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
    gpu_reserved_peak_bytes: int | None
    gpu_global_used_bytes: int | None
    gpu_global_free_bytes: int | None
    notes: str = ""
    cpu_percent: float | None = None
    cpu_count: int | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _pss_or_none(proc: psutil.Process) -> int | None:
    try:
        mi = proc.memory_full_info()
        value = getattr(mi, "pss", None)
        return None if value is None else int(value)
    except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
        return None


def _resolve_cuda_device(device: Any | None = None):
    """Bind GPU stats to the training device. Default is current_device (usually cuda:0)."""
    import torch

    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _gpu_fields(device: Any | None = None) -> dict[str, int | None]:
    out = {
        "gpu_alloc_bytes": None,
        "gpu_reserved_bytes": None,
        "gpu_alloc_peak_bytes": None,
        "gpu_reserved_peak_bytes": None,
        "gpu_global_used_bytes": None,
        "gpu_global_free_bytes": None,
    }
    try:
        import torch

        d = _resolve_cuda_device(device)
        if d is None:
            return out
        out["gpu_alloc_bytes"] = int(torch.cuda.memory_allocated(d))
        out["gpu_reserved_bytes"] = int(torch.cuda.memory_reserved(d))
        out["gpu_alloc_peak_bytes"] = int(torch.cuda.max_memory_allocated(d))
        out["gpu_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved(d))
        try:
            # On discrete GPUs this is board VRAM. On Jetson unified memory it
            # tracks the shared pool and can overlap host RSS; do not add the
            # two or treat gpu_global_* as independent VRAM.
            free, total = torch.cuda.mem_get_info(d)
            out["gpu_global_free_bytes"] = int(free)
            out["gpu_global_used_bytes"] = int(total - free)
        except Exception:
            pass
    except Exception:
        pass
    return out


def snapshot(
    phase: str,
    experience_index: int | None = None,
    notes: str = "",
    device: Any | None = None,
) -> ResourceSnapshot:
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
    gpu = _gpu_fields(device)
    cpu_percent = None
    cpu_count = None
    try:
        cpu_count = int(psutil.cpu_count() or 0) or None
        cpu_percent = float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
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
        cpu_percent=cpu_percent,
        cpu_count=cpu_count,
        **gpu,
    )


@dataclass
class SamplerPeaks:
    rss_peak_bytes: int = 0
    children_rss_peak_bytes: int = 0
    gpu_alloc_peak_bytes: int = 0
    gpu_reserved_peak_bytes: int = 0
    sample_count: int = 0
    phase: str = "idle"
    experience_index: int | None = None


class ResourceSampler:
    """Background RSS/allocator sampler. Peak is an observed lower bound.

    Phase tokens isolate in-flight samples: an old snapshot cannot update a
    newer phase's peaks. Attribution to the token captured at sample start is
    allowed.
    """

    def __init__(
        self,
        interval_s: float,
        writer: TextIO | None = None,
        device: Any | None = None,
    ) -> None:
        self.interval_s = interval_s
        self.writer = writer
        self.device = device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._token = 0
        self._phase = "idle"
        self._exp: int | None = None
        self._peaks: dict[int, SamplerPeaks] = {}
        self.peak_rss = 0
        self.peak_gpu_alloc = 0
        self.peak_gpu_reserved = 0
        self.phase_peak_rss = 0
        self.phase_peak_gpu_alloc = 0
        self.phase_peak_gpu_reserved = 0
        self.rows: list[ResourceSnapshot] = []

    def begin_phase(self, phase: str, experience_index: int | None) -> int:
        with self._lock:
            self._token += 1
            token = self._token
            self._phase = phase
            self._exp = experience_index
            self._peaks[token] = SamplerPeaks(phase=phase, experience_index=experience_index)
            self.phase_peak_rss = 0
            self.phase_peak_gpu_alloc = 0
            self.phase_peak_gpu_reserved = 0
            return token

    def set_phase(self, phase: str, experience_index: int | None) -> int:
        return self.begin_phase(phase, experience_index)

    def peaks_for(self, token: int | None) -> SamplerPeaks | None:
        if token is None:
            return None
        with self._lock:
            peaks = self._peaks.get(int(token))
            if peaks is None:
                return None
            return SamplerPeaks(
                rss_peak_bytes=peaks.rss_peak_bytes,
                children_rss_peak_bytes=peaks.children_rss_peak_bytes,
                gpu_alloc_peak_bytes=peaks.gpu_alloc_peak_bytes,
                gpu_reserved_peak_bytes=peaks.gpu_reserved_peak_bytes,
                sample_count=peaks.sample_count,
                phase=peaks.phase,
                experience_index=peaks.experience_index,
            )

    def ingest(self, snap: ResourceSnapshot, token: int) -> None:
        """Attribute a snapshot to the token captured before the sample."""
        rss = int(snap.proc_rss_bytes or 0)
        child = int(snap.children_rss_bytes or 0)
        gpu = int(snap.gpu_alloc_peak_bytes or snap.gpu_alloc_bytes or 0)
        reserved = int(snap.gpu_reserved_peak_bytes or snap.gpu_reserved_bytes or 0)
        with self._lock:
            peaks = self._peaks.get(int(token))
            if peaks is None:
                return
            peaks.rss_peak_bytes = max(peaks.rss_peak_bytes, rss)
            peaks.children_rss_peak_bytes = max(peaks.children_rss_peak_bytes, child)
            peaks.sample_count += 1
            if gpu:
                peaks.gpu_alloc_peak_bytes = max(peaks.gpu_alloc_peak_bytes, gpu)
            if reserved:
                peaks.gpu_reserved_peak_bytes = max(peaks.gpu_reserved_peak_bytes, reserved)
            self.peak_rss = max(self.peak_rss, rss + child)
            if gpu:
                self.peak_gpu_alloc = max(self.peak_gpu_alloc, gpu)
            if reserved:
                self.peak_gpu_reserved = max(self.peak_gpu_reserved, reserved)
            if token == self._token:
                self.phase_peak_rss = peaks.rss_peak_bytes + peaks.children_rss_peak_bytes
                self.phase_peak_gpu_alloc = peaks.gpu_alloc_peak_bytes
                self.phase_peak_gpu_reserved = peaks.gpu_reserved_peak_bytes

    def start(self) -> None:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
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
            with self._lock:
                token = self._token
                phase = self._phase
                exp = self._exp
            snap = snapshot(phase, exp, notes="interval", device=self.device)
            self.rows.append(snap)
            self.ingest(snap, token)
            if self.writer is not None:
                import csv

                row = snap.as_row()
                writer = csv.DictWriter(self.writer, fieldnames=list(row.keys()))
                if self.writer.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
                self.writer.flush()
            self._stop.wait(self.interval_s)


def reset_gpu_peak(device: Any | None = None) -> None:
    try:
        import torch

        d = _resolve_cuda_device(device)
        if d is not None:
            torch.cuda.reset_peak_memory_stats(d)
    except Exception:
        pass


def synchronize_gpu(device: Any | None = None) -> None:
    try:
        import torch

        d = _resolve_cuda_device(device)
        if d is not None:
            torch.cuda.synchronize(d)
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
