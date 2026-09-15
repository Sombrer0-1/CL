"""Per-phase GPU/host peaks. Polling RSS is an observed lower bound, not an instant max."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from orion_repro.memory.probe import reset_gpu_peak, snapshot, synchronize_gpu
from orion_repro.runner.artifacts import RunArtifacts


@dataclass
class PhaseRecord:
    phase: str
    experience_index: int | None
    start_monotonic_s: float
    end_monotonic_s: float
    duration_s: float
    allocated_current_bytes: int | None
    allocated_peak_bytes: int | None
    reserved_current_bytes: int | None
    reserved_peak_bytes: int | None
    quota_bytes: int | None
    external_reservation_bytes: int
    proc_rss_bytes: int | None
    children_rss_bytes: int | None
    proc_pss_bytes: int | None
    swap_used_bytes: int | None
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class PhaseRecorder:
    def __init__(
        self,
        arts: RunArtifacts,
        *,
        quota_bytes: int | None,
        reservation_getter: Callable[[], int],
        device: Any | None = None,
    ) -> None:
        self.arts = arts
        self.quota_bytes = quota_bytes
        self.reservation_getter = reservation_getter
        self.device = device
        self._open: dict[tuple[str, int | None], float] = {}
        self.last: PhaseRecord | None = None
        self.records: list[PhaseRecord] = []

    def begin(self, phase: str, k: int | None) -> None:
        synchronize_gpu(self.device)
        reset_gpu_peak(self.device)
        self._open[(phase, k)] = time.monotonic()

    def end(self, phase: str, k: int | None, notes: str = "") -> PhaseRecord:
        synchronize_gpu(self.device)
        end = time.monotonic()
        start = self._open.pop((phase, k))
        snap = snapshot(phase + "_end", k, device=self.device)
        rec = PhaseRecord(
            phase=phase,
            experience_index=k,
            start_monotonic_s=start,
            end_monotonic_s=end,
            duration_s=end - start,
            allocated_current_bytes=snap.gpu_alloc_bytes,
            allocated_peak_bytes=snap.gpu_alloc_peak_bytes,
            reserved_current_bytes=snap.gpu_reserved_bytes,
            reserved_peak_bytes=snap.gpu_reserved_peak_bytes,
            quota_bytes=self.quota_bytes,
            external_reservation_bytes=int(self.reservation_getter() or 0),
            proc_rss_bytes=snap.proc_rss_bytes,
            children_rss_bytes=snap.children_rss_bytes,
            proc_pss_bytes=snap.proc_pss_bytes,
            swap_used_bytes=snap.swap_used_bytes,
            notes=notes,
        )
        self.last = rec
        self.records.append(rec)
        self.arts.append_csv(self.arts._phases, rec.as_row())
        return rec
