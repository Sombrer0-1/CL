"""Process-local CUDA reservation that is not model or replay memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from orion_repro.memory.enforcement import install_device_quota
from orion_repro.memory.probe import synchronize_gpu


@dataclass
class QuotaRecord:
    mechanism: str
    limit_bytes: int
    fraction: float
    total_device_bytes: int
    scope: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReservationRecord:
    experience_index: int
    requested_bytes: int
    actual_tensor_bytes: int
    allocated_before: int | None
    allocated_after: int | None
    reserved_before: int | None
    reserved_after: int | None
    status: str
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_dyn_schedule(n_experiences: int, high_bytes: int) -> list[int]:
    """low/high/low in 3/3/3 blocks for NC-9. Other lengths keep equal thirds."""
    if n_experiences <= 0:
        raise ValueError("n_experiences must be positive")
    third = n_experiences // 3
    high_start = third
    high_end = 2 * third
    return [int(high_bytes) if high_start <= i < high_end else 0 for i in range(n_experiences)]


class ResourceEnvelope:
    def __init__(self, device) -> None:
        self.device = device
        self.quota: QuotaRecord | dict[str, Any] | None = None
        self._tensor = None
        self.reservation_bytes = 0

    def install_quota(self, budget: dict[str, Any], device=None) -> dict[str, Any] | None:
        target = device if device is not None else self.device
        raw = install_device_quota(budget, target)
        if raw is None:
            self.quota = None
            return None
        self.quota = QuotaRecord(**raw)
        return raw

    def transition(self, k: int, reservation_bytes: int) -> ReservationRecord:
        import torch

        requested = int(reservation_bytes)
        if requested < 0:
            raise ValueError("reservation_bytes must be >= 0")
        if requested and (self.device is None or getattr(self.device, "type", None) != "cuda"):
            raise RuntimeError("resource envelope reservation requires CUDA")
        allocated_before = reserved_before = None
        if torch.cuda.is_available() and getattr(self.device, "type", None) == "cuda":
            synchronize_gpu(self.device)
            allocated_before = int(torch.cuda.memory_allocated(self.device))
            reserved_before = int(torch.cuda.memory_reserved(self.device))
        status = "ok"
        notes = ""
        actual = int(self.reservation_bytes)
        unchanged = requested == int(self.reservation_bytes) and (
            requested == 0 or self._tensor is not None
        )
        if unchanged:
            notes = "unchanged"
        else:
            self._tensor = None
            self.reservation_bytes = 0
            actual = 0
        try:
            if requested and not unchanged:
                tensor = torch.empty(requested, dtype=torch.uint8, device=self.device)
                tensor.fill_(1)
                synchronize_gpu(self.device)
                self._tensor = tensor
                actual = int(tensor.numel()) * int(tensor.element_size())
                self.reservation_bytes = actual
        except torch.cuda.OutOfMemoryError:
            status = "resource_transition"
            notes = "allocation_failed"
            self._tensor = None
            self.reservation_bytes = 0
        allocated_after = reserved_after = None
        if torch.cuda.is_available() and getattr(self.device, "type", None) == "cuda":
            synchronize_gpu(self.device)
            allocated_after = int(torch.cuda.memory_allocated(self.device))
            reserved_after = int(torch.cuda.memory_reserved(self.device))
        return ReservationRecord(
            experience_index=int(k),
            requested_bytes=requested,
            actual_tensor_bytes=actual,
            allocated_before=allocated_before,
            allocated_after=allocated_after,
            reserved_before=reserved_before,
            reserved_after=reserved_after,
            status=status,
            notes=notes,
        )

    def close(self) -> None:
        """Release only the envelope tensor. Do not empty the global CUDA cache."""
        self._tensor = None
        self.reservation_bytes = 0
