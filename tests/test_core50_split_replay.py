from pathlib import Path

import torch
from torch.utils.data import Dataset

from orion_repro.benchmarks.core50_split import split_filelist_lines
from orion_repro.benchmarks.dev_split import FeedbackSourceError, resolve_feedback_source
from orion_repro.prefetch.dataloader import PrefetchingDataLoader, plan_index_batches
from orion_repro.runner.spec import RunSpec, load_yaml


class _LabeledToy(Dataset):
    def __init__(self, n: int, offset: int = 0) -> None:
        self.x = torch.arange(n).view(n, 1).float() + float(offset)
        self.targets = list(range(n))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], int(self.targets[i])


def test_core50_sequences_are_not_frame_mixed():
    lines = []
    for seq, n in (("s1/o1", 8), ("s2/o2", 8), ("s3/o3", 8), ("s4/o4", 8)):
        for i in range(n):
            lines.append(f"{seq}/C_{i:03d}.png 0\n")
    train, val, assignment = split_filelist_lines(
        lines, seed=17, val_fraction=0.25, experience_id=0
    )
    assert set(assignment.values()) <= {"train", "val"}
    assert "train" in assignment.values()
    assert "val" in assignment.values()
    train_seqs = {ln.split("/")[0] + "/" + ln.split("/")[1] for ln in train}
    val_seqs = {ln.split("/")[0] + "/" + ln.split("/")[1] for ln in val}
    assert not (train_seqs & val_seqs)
    for seq, role in assignment.items():
        bucket = train if role == "train" else val
        n = sum(1 for ln in bucket if ln.startswith(seq + "/"))
        assert n == 8


def test_core50_development_config_uses_val_feedback():
    spec = RunSpec.from_file(Path("configs/development/core50_nc_er_static.yaml"))
    assert spec.raw["controller"]["feedback_source"] == "development_val_seen"
    assert spec.raw["protocol_id"] == "development_val_diag_initial_v1"
    assert "acknowledged_official_test_exposure" not in spec.raw["dataset"]
    assert spec.raw["dataset"]["split_dir"].endswith("core50_nc_run0_dev_split_seed17")
    assert spec.raw["dataset"]["split_manifest"].endswith("core50_nc_run0_dev_split_seed17.json")


def test_core50_development_rejects_official_test():
    spec = load_yaml(Path("configs/development/core50_nc_er_static.yaml"))
    spec["controller"]["feedback_source"] = "official_test_seen"
    spec["dataset"]["acknowledged_official_test_exposure"] = True
    try:
        resolve_feedback_source(spec)
        raise AssertionError("expected FeedbackSourceError")
    except FeedbackSourceError:
        pass


def test_replay_dataloader_plan_is_deterministic():
    from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
    from avalanche.benchmarks.utils.utils import as_classification_dataset

    data = as_classification_dataset(_LabeledToy(24, 0))
    memory = as_classification_dataset(_LabeledToy(12, 100))

    def make_loader(seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        return ReplayDataLoader(
            data,
            memory,
            batch_size=4,
            batch_size_mem=4,
            shuffle=True,
            generator=g,
            num_workers=0,
            drop_last=False,
        )

    plan_a = plan_index_batches(make_loader(11))
    plan_b = plan_index_batches(make_loader(11))
    assert plan_a is not None and plan_b is not None
    idx_a, dataset, collate = plan_a
    idx_b, _, _ = plan_b
    assert idx_a == idx_b
    assert any(len(batch) > 4 for batch in idx_a), "replay concat batches should mix sources"
    serial = [collate([dataset[i] for i in batch]) for batch in idx_a]
    wrapped = PrefetchingDataLoader(
        make_loader(11), experience_id=1, config_version=0, max_depth=2
    )
    got = list(wrapped)
    assert wrapped.plan_mode == "main_thread_indices"
    assert wrapped.planned_index_batches == idx_a
    assert len(serial) == len(got)
    for a, b in zip(serial, got):
        torch.testing.assert_close(a[0], b[0])
        assert torch.equal(a[1], b[1])
