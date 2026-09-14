from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from orion_repro.benchmarks.dev_split import (
    FeedbackSourceError,
    build_cifar_dev_manifest,
    resolve_feedback_source,
)
from orion_repro.benchmarks.transforms import seeded_cifar_train
from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.rng import mix_seed
from orion_repro.runner.spec import RunSpec, SpecError, load_yaml


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def test_dev_split_is_stratified_and_covers_train():
    labels = [c for c in range(10) for _ in range(100)]
    payload = build_cifar_dev_manifest(dataset="cifar10", labels=labels, seed=17, val_fraction=0.2)
    assert payload["n_dev_train"] + payload["n_dev_val"] == 1000
    assert not (set(payload["train_indices"]) & set(payload["val_indices"]))
    for cls in range(10):
        n_tr = payload["per_class"][str(cls)]["train"]
        n_va = payload["per_class"][str(cls)]["val"]
        assert n_tr + n_va == 100
        assert n_va >= 1


def test_development_rejects_official_test_feedback():
    spec = load_yaml(Path("configs/development/cifar10_er_static.yaml"))
    spec["controller"]["feedback_source"] = "official_test_seen"
    try:
        resolve_feedback_source(spec)
        raise AssertionError("expected FeedbackSourceError")
    except FeedbackSourceError:
        pass


def test_unknown_feedback_source_is_rejected():
    spec = load_yaml(Path("configs/smoke.yaml"))
    spec["controller"]["feedback_source"] = "whatever"
    try:
        resolve_feedback_source(spec)
        raise AssertionError("expected FeedbackSourceError")
    except FeedbackSourceError:
        pass


def test_development_config_uses_val_feedback():
    spec = RunSpec.from_file(Path("configs/development/cifar10_er_static.yaml"))
    assert spec.raw["controller"]["feedback_source"] == "development_val_seen"
    assert spec.raw["dataset"]["split_manifest"].endswith("cifar10_dev_split_seed17.json")


class _SeededToy(Dataset):
    def __init__(self, n: int = 16) -> None:
        self.n = n
        self.images = [
            Image.fromarray(
                __import__("numpy").full((32, 32, 3), (i * 17) % 256, dtype="uint8")
            )
            for i in range(n)
        ]
        self.targets = list(range(n))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        x = seeded_cifar_train(
            self.images[i],
            aug_seed=123,
            sample_id=i,
            mean=CIFAR_MEAN,
            std=CIFAR_STD,
        )
        return x, self.targets[i]


def test_prefetch_planned_shuffle_is_deterministic():
    x = torch.arange(32).view(32, 1).float()
    y = torch.arange(32)
    ds = torch.utils.data.TensorDataset(x, y)

    def make_loader(seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=4, shuffle=True, generator=g, num_workers=0)

    from orion_repro.prefetch.dataloader import plan_index_batches

    plan_a = plan_index_batches(make_loader(7))
    plan_b = plan_index_batches(make_loader(7))
    assert plan_a is not None and plan_b is not None
    idx_a, dataset, collate = plan_a
    idx_b, _, _ = plan_b
    assert idx_a == idx_b
    serial = [collate([dataset[i] for i in batch]) for batch in idx_a]
    wrapped = PrefetchingDataLoader(make_loader(7), experience_id=0, config_version=0, max_depth=2)
    got = list(wrapped)
    assert wrapped.plan_mode == "main_thread_indices"
    assert wrapped.planned_index_batches == idx_a
    assert len(serial) == len(got)
    for a, b in zip(serial, got):
        torch.testing.assert_close(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_prefetch_matches_serial_with_seeded_augment():
    ds = _SeededToy(20)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    serial = [tuple(t.clone() if torch.is_tensor(t) else t for t in batch) for batch in loader]
    wrapped = PrefetchingDataLoader(loader, experience_id=0, config_version=0, max_depth=2)
    prefetched = [tuple(t.clone() if torch.is_tensor(t) else t for t in batch) for batch in wrapped]
    assert len(serial) == len(prefetched)
    for a, b in zip(serial, prefetched):
        torch.testing.assert_close(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_prefetch_queue_matches_serial_plan():
    x = torch.arange(32).view(32, 1).float()
    y = torch.arange(32)
    ds = torch.utils.data.TensorDataset(x, y)

    def make_loader(seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=4, shuffle=True, generator=g, num_workers=0)

    serial = PrefetchingDataLoader(
        make_loader(7), experience_id=0, config_version=0, max_depth=2, use_queue=False
    )
    queued = PrefetchingDataLoader(
        make_loader(7), experience_id=0, config_version=0, max_depth=2, use_queue=True
    )
    a = list(serial)
    b = list(queued)
    assert serial.plan_mode == "main_thread_indices_serial"
    assert queued.plan_mode == "main_thread_indices"
    assert serial.planned_index_batches == queued.planned_index_batches
    assert len(a) == len(b)
    for u, v in zip(a, b):
        torch.testing.assert_close(u[0], v[0])
        assert torch.equal(u[1], v[1])
    assert mix_seed(17, 4, "x") == mix_seed(17, 4, "x")
    assert mix_seed(17, 4) != mix_seed(17, 5)


def test_formal_spec_rejects_null_provenance():
    spec = load_yaml(Path("configs/smoke.yaml"))
    spec["phase"] = "formal"
    try:
        from orion_repro.runner.spec import validate_mapping

        validate_mapping(spec, require_provenance=True)
        raise AssertionError("expected SpecError")
    except SpecError:
        pass
