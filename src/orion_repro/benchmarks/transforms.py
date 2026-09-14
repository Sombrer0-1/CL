"""Deterministic per-sample CIFAR augmentation (PLAN §7.5).

RandomCrop+Flip is a function of (augmentation_seed, sample_id), not of which
thread consumes the global torch RNG. Prefetch on/off can therefore match.
"""

from __future__ import annotations

import hashlib

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from orion_repro.rng import mix_seed, torch_generator


def seeded_cifar_train(
    img: Image.Image,
    *,
    aug_seed: int,
    sample_id: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_size: int = 32,
    padding: int = 4,
) -> torch.Tensor:
    g = torch_generator(mix_seed(int(aug_seed), int(sample_id), "cifar_train_v1"))
    img = TF.pad(img, padding, fill=0, padding_mode="constant")
    w, h = img.size
    max_i = h - crop_size
    max_j = w - crop_size
    top = int(torch.randint(0, max_i + 1, (1,), generator=g).item())
    left = int(torch.randint(0, max_j + 1, (1,), generator=g).item())
    img = TF.crop(img, top, left, crop_size, crop_size)
    if float(torch.rand(1, generator=g).item()) < 0.5:
        img = TF.hflip(img)
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, mean, std)


def cifar_eval(
    img: Image.Image,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(img), mean, std)


CORE50_MEAN = (0.485, 0.456, 0.406)
CORE50_STD = (0.229, 0.224, 0.225)


def _image_content_id(img: Image.Image) -> int:
    digest = hashlib.sha256(img.convert("RGB").tobytes()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def seeded_core50_train(img: Image.Image, *, aug_seed: int) -> torch.Tensor:
    """ImageNet-normalized CORe50 mini train transform; flip from (aug_seed, content)."""
    sid = _image_content_id(img)
    g = torch_generator(mix_seed(int(aug_seed), sid, "core50_hflip_v1"))
    if float(torch.rand(1, generator=g).item()) < 0.5:
        img = TF.hflip(img)
    return TF.normalize(TF.to_tensor(img), CORE50_MEAN, CORE50_STD)


def seeded_core50_eval(img: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(img), CORE50_MEAN, CORE50_STD)


class SeededCore50Train:
    """Picklable train transform; flip depends on (aug_seed, image bytes)."""

    def __init__(self, aug_seed: int) -> None:
        self.aug_seed = int(aug_seed)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return seeded_core50_train(img, aug_seed=self.aug_seed)
