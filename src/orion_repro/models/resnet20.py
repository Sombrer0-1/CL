"""CIFAR ResNet-20 (He et al., option A).

Paper §5.1 specifies ResNet-20 [38] without the CIFAR variant details.
This module follows He et al. 2016 §4.2: n=3, channels 16-32-64, identity
shortcuts with zero-pad (option A). Not torchvision ImageNet ResNet-18.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _kaiming_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight)


class IdentityPadShortcut(nn.Module):
    """Option A: spatial stride-2 subsample and channel zero-pad."""

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.planes = planes

    def forward(self, x):
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.planes // 4, self.planes // 4),
            "constant",
            0,
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = IdentityPadShortcut(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, num_blocks: list[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_kaiming_init)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def lower(self, x):
        """Stem + layer1 + layer2; latent for reconstructed Latent Replay (A11)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return self.layer2(out)

    def upper(self, latent):
        out = self.layer3(latent)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and bool(getattr(self, "_latent_freeze_lower", False)):
            for module in (self.conv1, self.bn1, self.layer1, self.layer2):
                module.eval()
        return self

    def forward(self, x):
        replay_z = getattr(self, "_latent_replay_z", None)
        freeze_lower = bool(getattr(self, "_latent_freeze_lower", False))
        if freeze_lower:
            with torch.no_grad():
                z = self.lower(x)
            z = z.detach()
        else:
            z = self.lower(x)
        if replay_z is not None:
            z = torch.cat([z, replay_z], dim=0)
        return self.upper(z)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet([3, 3, 3], num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
