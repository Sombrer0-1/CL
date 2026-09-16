"""Per-phase GPU/host peaks. Polling RSS is an observed lower bound, not an instant max."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from orion_repro.memory.probe import ResourceSampler, reset_gpu_peak, snapshot, synchronize_gpu
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
    sampled_rss_peak_bytes: int | None = None
    sampled_children_rss_peak_bytes: int | None = None
    sample_interval_s: float | None = None
    sample_count: int = 0
    missing_reason: str = ""
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
        sampler: ResourceSampler | None = None,
    ) -> None:
        self.arts = arts
        self.quota_bytes = quota_bytes
        self.reservation_getter = reservation_getter
        self.device = device
        self.sampler = sampler
        self._open: dict[tuple[str, int | None], tuple[float, int | None]] = {}
        self.last: PhaseRecord | None = None
        self.records: list[PhaseRecord] = []

    def set_sampler(self, sampler: ResourceSampler | None) -> None:
        self.sampler = sampler

    def begin(self, phase: str, k: int | None) -> None:
        if self._open:
            raise RuntimeError("phase overlap would reset another phase's GPU peaks")
        token = None
        if self.sampler is not None:
            token = self.sampler.begin_phase(phase, k)
        synchronize_gpu(self.device)
        reset_gpu_peak(self.device)
        self._open[(phase, k)] = (time.monotonic(), token)

    def end(self, phase: str, k: int | None, notes: str = "") -> PhaseRecord:
        synchronize_gpu(self.device)
        end = time.monotonic()
        start, token = self._open.pop((phase, k))
        snap = snapshot(phase + "_end", k, device=self.device)
        sampled_rss = sampled_child = sample_count = None
        missing = "sampler_not_attached"
        interval = None
        if self.sampler is not None:
            interval = float(self.sampler.interval_s)
            peaks = self.sampler.peaks_for(token)
            if peaks is None:
                sampled_rss = 0
                sampled_child = 0
                sample_count = 0
                missing = "no_samples_for_token"
            else:
                sampled_rss = int(peaks.rss_peak_bytes)
                sampled_child = int(peaks.children_rss_peak_bytes)
                sample_count = int(peaks.sample_count)
                missing = "" if sample_count else "no_interval_samples"
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
            sampled_rss_peak_bytes=sampled_rss,
            sampled_children_rss_peak_bytes=sampled_child,
            sample_interval_s=interval,
            sample_count=int(sample_count or 0),
            missing_reason=missing,
            notes=notes,
        )
        self.last = rec
        self.records.append(rec)
        self.arts.append_csv(self.arts._phases, rec.as_row())
        return rec

    def end_failed(self, phase: str, k: int | None) -> PhaseRecord | None:
        """Capture the failed phase before cleanup; never substitute an older phase."""
        if (phase, k) in self._open:
            return self.end(phase, k, notes="failed")
        if self.last is not None and (self.last.phase, self.last.experience_index) == (phase, k):
            return self.last
        return None
