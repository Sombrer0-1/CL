"""Fixed development split from official training data (PLAN §4.3, A06).

CIFAR: stratified by class, split_seed=17. Official test is not used.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion_repro.rng import mix_seed

ROOT = Path(__file__).resolve().parents[3]

KNOWN_FEEDBACK_SOURCES = {
    "official_test_seen",
    "development_val_seen",
    "heldout_train_control",
}

DEV_SPLIT_SEED = 17
DEV_VAL_FRACTION = 0.20


class FeedbackSourceError(ValueError):
    pass


def resolve_feedback_source(spec: dict[str, Any]) -> str:
    src = spec["controller"].get("feedback_source")
    if src not in KNOWN_FEEDBACK_SOURCES:
        raise FeedbackSourceError(
            f"unknown feedback_source {src!r}; known={sorted(KNOWN_FEEDBACK_SOURCES)}"
        )
    phase = str(spec.get("phase", ""))
    if phase == "development" and src == "official_test_seen":
        raise FeedbackSourceError(
            "development phase cannot use official_test_seen (PLAN §4.3). "
            "Use development_val_seen. Historical runs that used the official "
            "test remain exploratory and test-exposed. CORe50 uses sequence-"
            "grouped filelists under data/processed/core50_*_dev_split_seed17."
        )
    if phase == "formal" and src == "development_val_seen":
        raise FeedbackSourceError(
            "formal phase cannot use development_val_seen; freeze a paper_feedback "
            "or validation_feedback protocol first"
        )
    return str(src)


def _stratified_indices(
    labels: list[int],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_class[int(y)].append(int(idx))
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in sorted(by_class):
        members = list(by_class[cls])
        members.sort()
        # Deterministic permutation independent of PYTHONHASHSEED.
        order = sorted(
            range(len(members)),
            key=lambda i: mix_seed(seed, cls, members[i], "dev_split_v1"),
        )
        shuffled = [members[i] for i in order]
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])
    train_idx.sort()
    val_idx.sort()
    overlap = set(train_idx) & set(val_idx)
    if overlap:
        raise RuntimeError(f"dev split leaked {len(overlap)} indices")
    if set(train_idx) | set(val_idx) != set(range(len(labels))):
        raise RuntimeError("dev split does not cover the official train set")
    return train_idx, val_idx


def build_cifar_dev_manifest(
    *,
    dataset: str,
    labels: list[int],
    seed: int = DEV_SPLIT_SEED,
    val_fraction: float = DEV_VAL_FRACTION,
) -> dict[str, Any]:
    train_idx, val_idx = _stratified_indices(labels, seed=seed, val_fraction=val_fraction)
    per_class: dict[str, dict[str, int]] = {}
    for cls in sorted(set(labels)):
        n_tr = sum(1 for i in train_idx if int(labels[i]) == cls)
        n_va = sum(1 for i in val_idx if int(labels[i]) == cls)
        per_class[str(cls)] = {"train": n_tr, "val": n_va}
    return {
        "dataset": dataset,
        "source": "official_train_only",
        "split_seed": seed,
        "val_fraction": val_fraction,
        "stratify": "class",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "n_official_train": len(labels),
        "n_dev_train": len(train_idx),
        "n_dev_val": len(val_idx),
        "n_official_test_unused": True,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "per_class": per_class,
        "note": (
            "PLAN §4.3 development split. Indices are into the official train set. "
            "Official test images are not included. Changing split_seed or "
            "val_fraction requires a new manifest version; it does not un-expose "
            "historical official-test runs."
        ),
    }


def write_cifar_dev_manifest(dataset: str, root: Path, out_path: Path) -> dict[str, Any]:
    from torchvision.datasets import CIFAR10, CIFAR100

    cls = CIFAR10 if dataset == "cifar10" else CIFAR100
    train = cls(root=str(root), train=True, download=False)
    labels = [int(y) for y in train.targets]
    payload = build_cifar_dev_manifest(dataset=dataset, labels=labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_dev_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train_indices", "val_indices", "split_seed", "dataset"):
        if key not in payload:
            raise ValueError(f"{path} missing {key}")
    return payload
