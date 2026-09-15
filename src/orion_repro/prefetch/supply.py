"""Consumption hashing and supply-time accounting for the IO scenario."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def extract_xy(batch: Any) -> tuple[Any, Any] | tuple[None, None]:
    if hasattr(batch, "x") and hasattr(batch, "y"):
        return batch.x, batch.y
    if isinstance(batch, dict) and "x" in batch and "y" in batch:
        return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    return None, None


def tensor_sha256(value: Any) -> str | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def update_rolling(current: Any, *parts: str | None) -> None:
    for part in parts:
        if part:
            current.update(part.encode("ascii"))
