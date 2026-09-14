"""GEM that can be enabled after experience 0 (A10).

Avalanche GEMPlugin indexes memory_x[t] for every t < train_exp_counter.
If the plugin was paused, those keys are missing. PLAN §7.2: do not
rebuild discarded history from disk; skip missing experiences.
"""

from __future__ import annotations

import torch
from avalanche.models import avalanche_forward
from avalanche.training.plugins.gem import GEMPlugin


class SparseGEMPlugin(GEMPlugin):
    """GEM that skips missing experience keys and counts auxiliary forwards."""

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        super().__init__(patterns_per_experience, memory_strength)
        self.auxiliary_visits = 0
        self.projection_count = 0
        self.qp_failures = 0

    def before_training_exp(self, strategy, **kwargs):
        self.auxiliary_visits = 0
        self.projection_count = 0

    def before_training_iteration(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter <= 0:
            return
        present = [
            t for t in range(strategy.clock.train_exp_counter) if t in self.memory_x
        ]
        if not present:
            self.G = torch.empty(0)
            return
        G = []
        strategy.model.train()
        for t in present:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            xref = self.memory_x[t].to(strategy.device)
            yref = self.memory_y[t].to(strategy.device)
            self.auxiliary_visits += int(xref.shape[0])
            out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
            loss = strategy._criterion(out, yref)
            loss.backward()
            G.append(
                torch.cat(
                    [
                        (
                            p.grad.flatten()
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=strategy.device)
                        )
                        for p in strategy.model.parameters()
                    ],
                    dim=0,
                )
            )
        self.G = torch.stack(G)

    def resize(self, patterns_per_experience: int) -> None:
        """Shrink stored patterns; expansion does not restore discarded samples."""
        self.patterns_per_experience = int(patterns_per_experience)
        keep = max(0, self.patterns_per_experience)
        for t in list(self.memory_x.keys()):
            n = int(self.memory_x[t].shape[0])
            if keep == 0:
                del self.memory_x[t]
                del self.memory_y[t]
                del self.memory_tid[t]
                continue
            if n > keep:
                self.memory_x[t] = self.memory_x[t][:keep].clone()
                self.memory_y[t] = self.memory_y[t][:keep].clone()
                self.memory_tid[t] = self.memory_tid[t][:keep].clone()

    def after_backward(self, strategy, **kwargs):
        if self.G is None or self.G.numel() == 0:
            return
        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )
            if (torch.mv(self.G, g) < 0).any():
                self.projection_count += 1
        return super().after_backward(strategy, **kwargs)

    def solve_quadprog(self, g):
        """Avalanche GEM + qpsolvers can return None; do not crash the run.

        Infeasible projection leaves the current gradient unchanged and
        increments qp_failures. This is a platform compatibility guard, not
        the original GEM paper solver.
        """
        import numpy as np
        import qpsolvers

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        if v is None:
            self.qp_failures = int(getattr(self, "qp_failures", 0)) + 1
            return g.detach().to("cpu").float()
        v_star = np.dot(v, memories_np) + gradient_np
        return torch.from_numpy(v_star).float()


class AdaptiveGEMPlugin(SparseGEMPlugin):
    """Base-GEM adapter used by Orion; same missing-history and QP guards."""
