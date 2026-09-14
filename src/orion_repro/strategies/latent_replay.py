"""Latent Replay reconstructed for CIFAR ResNet-20 (A11).

Not the authors' Orion code. Follows Pellegrini et al. / Avalanche AR1:
store layer2 activations, freeze the stem after the first experience,
train the upper block on new latents plus cached latents. No backprop
through stored latents or the frozen stem.
"""

from __future__ import annotations

import torch
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader

from orion_repro.models.resnet20 import CifarResNet


class LatentReplayPlugin(SupervisedPlugin):
    def __init__(self, mem_size: int = 200, replay_batch: int | None = None) -> None:
        super().__init__()
        self.mem_size = int(mem_size)
        self.replay_batch = replay_batch
        self.buffer_z: torch.Tensor | None = None
        self.buffer_y: torch.Tensor | None = None
        self._frozen = False

    def before_training_exp(self, strategy, *args, **kwargs) -> None:
        self.replay_visits = 0
        self.auxiliary_visits = 0
        exp_id = int(getattr(getattr(strategy, "clock", None), "train_exp_counter", 0) or 0)
        model = strategy.model
        if not isinstance(model, CifarResNet):
            raise TypeError("LatentReplayPlugin expects CifarResNet")
        if exp_id > 0 and not self._frozen:
            self._frozen = True
        model._latent_freeze_lower = self._frozen
        model._latent_replay_z = None
        if self._frozen:
            self._freeze_lower(model)

    def before_forward(self, strategy, *args, **kwargs) -> None:
        model = strategy.model
        n_replay = self.replay_batch if self.replay_batch is not None else int(strategy.train_mb_size)
        if self.buffer_z is None or self.buffer_z.shape[0] == 0 or n_replay <= 0:
            model._latent_replay_z = None
            return
        n = min(n_replay, int(self.buffer_z.shape[0]))
        self.replay_visits += n
        idx = torch.randint(0, self.buffer_z.shape[0], (n,), device=self.buffer_z.device)
        z = self.buffer_z[idx].to(strategy.device, non_blocking=False)
        y = self.buffer_y[idx].to(strategy.device, non_blocking=False)
        model._latent_replay_z = z
        y_all = torch.cat([strategy.mb_y, y], dim=0)
        strategy.mbatch[1] = y_all
        if len(strategy.mbatch) > 2:
            tid = strategy.mbatch[2]
            extra = torch.zeros(n, dtype=tid.dtype, device=tid.device)
            strategy.mbatch[2] = torch.cat([tid, extra], dim=0)

    def after_training_iteration(self, strategy, *args, **kwargs) -> None:
        strategy.model._latent_replay_z = None

    def after_training_exp(self, strategy, *args, **kwargs) -> None:
        model: CifarResNet = strategy.model
        dataset = strategy.experience.dataset
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        was_training = model.training
        model.eval()
        zs = []
        ys = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                x = x.to(strategy.device)
                self.auxiliary_visits += int(x.shape[0])
                z = model.lower(x).detach().cpu()
                zs.append(z)
                ys.append(y.cpu() if torch.is_tensor(y) else torch.as_tensor(y))
                if sum(t.shape[0] for t in zs) >= self.mem_size:
                    break
        model.train(was_training)
        if not zs:
            return
        z_cat = torch.cat(zs, dim=0)[: self.mem_size]
        y_cat = torch.cat(ys, dim=0)[: self.mem_size]
        if self.buffer_z is None:
            self.buffer_z, self.buffer_y = z_cat, y_cat
        else:
            self.buffer_z = torch.cat([self.buffer_z, z_cat], dim=0)
            self.buffer_y = torch.cat([self.buffer_y, y_cat], dim=0)
            if self.buffer_z.shape[0] > self.mem_size:
                perm = torch.randperm(self.buffer_z.shape[0])[: self.mem_size]
                self.buffer_z = self.buffer_z[perm]
                self.buffer_y = self.buffer_y[perm]

    @staticmethod
    def _freeze_lower(model: CifarResNet) -> None:
        for module in (model.conv1, model.bn1, model.layer1, model.layer2):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
