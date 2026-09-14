"""Independent seed streams (PLAN §4.3 / §7.5)."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def mix_seed(*parts: int | str) -> int:
    """Stable 63-bit seed from integer/string parts. Not Python hash()."""
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) % (2**63))
    return g


def seed_python_and_numpy(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))


def seed_torch_global(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_streams(spec: dict[str, Any]) -> dict[str, int]:
    """Consume and record the four named seeds. Model seed still seeds global RNGs.

    Replay/augmentation are independent streams: they must not be silently ignored.
    Global torch RNG is set from the model seed so init is reproducible; per-sample
    augmentation uses mix_seed(augmentation, sample_id) instead of the global RNG.
    """
    seeds = spec["seeds"]
    required = ("model", "stream", "replay", "augmentation")
    missing = [k for k in required if k not in seeds]
    if missing:
        raise ValueError(f"seeds missing {missing}")
    out = {k: int(seeds[k]) for k in required}
    seed_python_and_numpy(out["model"])
    seed_torch_global(out["model"])
    return out
