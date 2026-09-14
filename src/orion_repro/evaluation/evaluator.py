"""No-grad evaluation that restores train mode and RNG (PLAN §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class DomainAccuracy:
    eval_domain: int
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return float("nan")
        return self.correct / self.total


def snapshot_rng(device: torch.device | None = None) -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _unpack(batch: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) < 2:
        raise ValueError(f"unexpected batch structure: {type(batch)} len={len(batch)}")
    return batch[0], batch[1]


@torch.no_grad()
def evaluate_domains(
    model: torch.nn.Module,
    domains: Iterable[tuple[int, Any]],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> list[DomainAccuracy]:
    was_training = model.training
    rng = snapshot_rng()
    model.eval()
    rows: list[DomainAccuracy] = []
    try:
        for domain_id, dataset in domains:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )
            correct = 0
            total = 0
            for batch in loader:
                x, y = _unpack(batch)
                x = x.to(device, non_blocking=False)
                y = y.to(device, non_blocking=False)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
            rows.append(DomainAccuracy(eval_domain=int(domain_id), correct=correct, total=total))
    finally:
        model.train(was_training)
        restore_rng(rng)
    return rows
