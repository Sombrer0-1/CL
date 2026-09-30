"""Board-level controller observation for fullmem_v4 O-recon.

M_max is actual MemTotal. M_t is the training-window peak of
MemTotal-MemAvailable. Host RSS and CUDA allocated are recorded separately
and must not be summed into the board total.
"""

from __future__ import annotations

from typing import Any

# PDF data representation for the 32x32 RGB samples used by this stage.
RAW_RGB32_UINT8_BYTES = 3 * 32 * 32


def board_used_bytes(mem_total_bytes: int | None, mem_available_bytes: int | None) -> int:
    if mem_total_bytes is None or mem_available_bytes is None:
        return 0
    return max(0, int(mem_total_bytes) - int(mem_available_bytes))


def paper_data_representation(
    *,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    sample_bytes: int = 1,
) -> dict[str, Any]:
    """Fixed per-sample bytes from the stored data representation.

    This is the O-recon mapping for Eq. (3)/(4). Fitted peak slopes stay in
    O-eng cost diagnosis and must not replace these values.
    """
    frame = int(channels) * int(height) * int(width) * int(sample_bytes)
    return {
        "m_batch": float(frame),
        "m_frame": float(frame),
        "bytes_per_sample": frame,
        "unit": "bytes",
        "representation": "uint8_rgb_32x32",
        "note": "paper data-representation bytes; activations/gradients/workspace are recorded separately",
    }


def byte_space_initial_budgets(
    new_batch: int,
    replay_capacity: int,
    *,
    m_batch: float,
    m_frame: float,
) -> tuple[float, float]:
    if m_batch <= 0 or m_frame <= 0:
        raise ValueError("m_batch and m_frame must be positive")
    return float(new_batch) * float(m_batch), float(replay_capacity) * float(m_frame)


def resolve_controller_memory_bytes(
    resource_name: str,
    *,
    board_used_bytes_value: int | None = None,
    gpu_peak_bytes: int | None = None,
    host_rss_bytes: int | None = None,
) -> int:
    name = str(resource_name or "")
    if name in {"board", "shared_pool"}:
        if board_used_bytes_value is None:
            raise ValueError("controlled_resource=board requires MemTotal-MemAvailable peak")
        return int(board_used_bytes_value)
    if name == "host":
        return int(host_rss_bytes or 0)
    if name == "device":
        return int(gpu_peak_bytes or 0)
    raise ValueError(f"unknown controlled_resource {resource_name!r}")
