"""AGEM with dynamic per-experience capacity and auxiliary visit counts."""

from __future__ import annotations

import warnings

import torch
from avalanche.training.plugins.agem import AGEMPlugin
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader


class AdaptiveAGEMPlugin(AGEMPlugin):
    """Official AGEM projection plus resize that does not restore discarded samples."""

    def __init__(self, patterns_per_experience: int, sample_size: int):
        super().__init__(patterns_per_experience, sample_size)
        self.auxiliary_visits = 0
        self.projection_count = 0

    def before_training_exp(self, strategy, **kwargs):
        self.auxiliary_visits = 0
        self.projection_count = 0

    def before_training_iteration(self, strategy, **kwargs):
        super().before_training_iteration(strategy, **kwargs)
        if len(self.buffers) > 0:
            mb = getattr(self, "_last_memory_mb", None)
            if mb is None:
                self.auxiliary_visits += int(self.sample_size)
            else:
                self.auxiliary_visits += int(mb[0].shape[0])

    def sample_from_memory(self):
        mb = super().sample_from_memory()
        self._last_memory_mb = mb
        return mb

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        if len(self.buffers) > 0:
            current = torch.cat(
                [
                    (
                        p.grad.view(-1)
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for n, p in strategy.model.named_parameters()
                ]
            )
            if torch.dot(current, self.reference_gradients) < 0:
                self.projection_count += 1
        return super().after_backward(strategy, **kwargs)

    def resize(
        self,
        patterns_per_experience: int,
        sample_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        self.patterns_per_experience = int(patterns_per_experience)
        if sample_size is not None:
            self.sample_size = max(1, int(sample_size)) if self.patterns_per_experience > 0 else 0
        if self.patterns_per_experience <= 0:
            self.buffers = []
            self.buffer_dataloader = None
            self.buffer_dliter = iter([])
            self.reference_gradients = torch.empty(0)
            return
        new_buffers = []
        for buf in self.buffers:
            n = len(buf)
            if n > self.patterns_per_experience:
                new_buffers.append(buf.subset(list(range(self.patterns_per_experience))))
            else:
                new_buffers.append(buf)
        self.buffers = new_buffers
        if not self.buffers:
            self.buffer_dataloader = None
            self.buffer_dliter = iter([])
            return
        if num_workers > 0:
            warnings.warn("Num workers > 0 is known to cause heavy slowdowns in AGEM.")
        n_buf = len(self.buffers)
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=max(1, self.sample_size // n_buf),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
        self.buffer_dliter = iter(self.buffer_dataloader)
