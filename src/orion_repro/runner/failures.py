"""Failure context must distinguish completed training from completed evaluation."""

from __future__ import annotations

import re
from typing import Any

_ALLOCATE_RE = re.compile(
    r"Tried to allocate\s+([0-9]*\.?[0-9]+)\s*(KiB|MiB|GiB|B)\b",
    re.IGNORECASE,
)
_UNIT_BYTES = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3}


def parse_cuda_oom_request_bytes(message: str) -> int | None:
    match = _ALLOCATE_RE.search(message or "")
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    return int(round(value * _UNIT_BYTES[unit]))


def pathological_memory_value(nbytes: int | None, *, device_total_bytes: int | None = None) -> bool:
    """Huge non-Torch values in OOM text must not enter resource statistics."""
    if nbytes is None:
        return False
    if nbytes < 0:
        return True
    if device_total_bytes is not None and nbytes > 2 * int(device_total_bytes):
        return True
    # 1 TiB is already far beyond this platform's GPU.
    return nbytes >= 1024**4


def failure_metadata(
    spec: dict[str, Any],
    *,
    phase: str,
    experience: int | None,
    trained: int,
    evaluated: int,
    request_bytes: int | None = None,
    quota_bytes: int | None = None,
    unannounced_transition: bool | None = None,
    allocated_peak_bytes: int | None = None,
    reserved_peak_bytes: int | None = None,
    external_reservation_bytes: int | None = None,
) -> dict[str, Any]:
    if pathological_memory_value(request_bytes) or pathological_memory_value(allocated_peak_bytes):
        request_bytes = None if pathological_memory_value(request_bytes) else request_bytes
        allocated_peak_bytes = None if pathological_memory_value(allocated_peak_bytes) else allocated_peak_bytes
        reserved_peak_bytes = None if pathological_memory_value(reserved_peak_bytes) else reserved_peak_bytes
    return {
        "failure_phase": phase,
        "failure_experience": experience,
        "n_experiences_trained": trained,
        "n_experiences_run": evaluated,
        "n_experiences_evaluated": evaluated,
        "phase": spec["phase"],
        "dataset": spec["dataset"]["name"],
        "method_id": spec["method_id"],
        "eval_batch": spec["training"]["eval_batch"],
        "budget_enforcement": spec["budget"]["enforcement"],
        "budget_limit_bytes": spec["budget"].get("limit_bytes"),
        "request_bytes": request_bytes,
        "quota_bytes": quota_bytes if quota_bytes is not None else spec["budget"].get("limit_bytes"),
        "unannounced_transition": unannounced_transition,
        "allocated_peak_bytes": allocated_peak_bytes,
        "reserved_peak_bytes": reserved_peak_bytes,
        "external_reservation_bytes": external_reservation_bytes,
    }
