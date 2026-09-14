"""CIFAR / CORe50 / Endless benchmark construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from orion_repro.benchmarks.dev_split import (
    DEV_SPLIT_SEED,
    DEV_VAL_FRACTION,
    load_dev_split,
    resolve_feedback_source,
)
from orion_repro.rng import mix_seed
from orion_repro.benchmarks.transforms import (
    SeededCore50Train,
    cifar_eval,
    seeded_cifar_train,
    seeded_core50_eval,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CIFAR_MEAN = CIFAR10_MEAN
CIFAR_STD = CIFAR10_STD

# Kept for smoke that still uses Avalanche SplitCIFAR official transforms.
# Development and prefetch-paired runs use per-sample seeded transforms instead.
CIFAR_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)
CIFAR_EVAL = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)
CIFAR100_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)
CIFAR100_EVAL = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)

ROOT = Path(__file__).resolve().parents[3]


class SeededCIFARSubset(Dataset):
    """Official-train subset with per-sample seeded augmentation.

    sample_id is the original CIFAR train index, so prefetch threads cannot
    change the crop/flip of a given image.
    """

    def __init__(
        self,
        base: Dataset,
        indices: list[int],
        *,
        train: bool,
        aug_seed: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.base = base
        self.indices = [int(i) for i in indices]
        self.train = bool(train)
        self.aug_seed = int(aug_seed)
        self.mean = mean
        self.std = std
        targets = getattr(base, "targets")
        self.targets = [int(targets[i]) for i in self.indices]
        self.sample_ids = list(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        orig = self.indices[i]
        img, y = self.base[orig]
        if self.train:
            tensor = seeded_cifar_train(
                img, aug_seed=self.aug_seed, sample_id=orig, mean=self.mean, std=self.std
            )
        else:
            tensor = cifar_eval(img, mean=self.mean, std=self.std)
        return tensor, int(y)


def build_benchmark(spec: dict[str, Any]):
    ds = spec["dataset"]
    name = ds["name"]
    root = Path(ds["root"])
    if not root.is_absolute():
        root = ROOT / root
    feedback = resolve_feedback_source(spec)
    if name.startswith("core50"):
        return _build_core50(spec, root, feedback)
    if name.startswith("endless"):
        return _build_endless(spec, root, feedback)
    if name in {"splitcifar10", "splitcifar100"}:
        return _build_cifar(spec, root, feedback)
    raise ValueError(f"unsupported dataset {name}")


def _build_cifar(spec: dict[str, Any], root: Path, feedback: str):
    from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark

    ds = spec["dataset"]
    name = ds["name"]
    n_exp = int(ds["n_experiences"])
    stream_seed = int(spec["seeds"]["stream"])
    aug_seed = int(spec["seeds"]["augmentation"])
    shuffle = bool(ds.get("shuffle", True))
    fixed = ds.get("fixed_class_order")
    mean = CIFAR10_MEAN if name == "splitcifar10" else CIFAR100_MEAN
    std = CIFAR10_STD if name == "splitcifar10" else CIFAR100_STD
    cls = CIFAR10 if name == "splitcifar10" else CIFAR100
    official_train = cls(root=str(root), train=True, download=False)
    official_test = cls(root=str(root), train=False, download=False)

    if feedback == "development_val_seen":
        manifest_rel = ds.get("split_manifest")
        if not manifest_rel:
            raise ValueError("development_val_seen requires dataset.split_manifest")
        manifest_path = Path(manifest_rel)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        split = load_dev_split(manifest_path)
        expected = "cifar10" if name == "splitcifar10" else "cifar100"
        if split["dataset"] != expected:
            raise ValueError(f"split manifest dataset {split['dataset']} != {expected}")
        train_ds = SeededCIFARSubset(
            official_train,
            split["train_indices"],
            train=True,
            aug_seed=aug_seed,
            mean=mean,
            std=std,
        )
        eval_ds = SeededCIFARSubset(
            official_train,
            split["val_indices"],
            train=False,
            aug_seed=aug_seed,
            mean=mean,
            std=std,
        )
    elif feedback in {"official_test_seen", "heldout_train_control"}:
        if feedback == "heldout_train_control":
            raise ValueError("heldout_train_control is reserved for formal validation_feedback")
        all_train = list(range(len(official_train)))
        all_test = list(range(len(official_test)))
        train_ds = SeededCIFARSubset(
            official_train,
            all_train,
            train=True,
            aug_seed=aug_seed,
            mean=mean,
            std=std,
        )
        eval_ds = SeededCIFARSubset(
            official_test,
            all_test,
            train=False,
            aug_seed=aug_seed,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(f"unhandled feedback_source {feedback}")

    return nc_benchmark(
        train_dataset=train_ds,
        test_dataset=eval_ds,
        n_experiences=n_exp,
        task_labels=False,
        seed=stream_seed if fixed is None else None,
        shuffle=shuffle if fixed is None else False,
        fixed_class_order=list(fixed) if fixed is not None else None,
        train_transform=None,
        eval_transform=None,
    )


def _build_core50(spec: dict[str, Any], root: Path, feedback: str):
    from avalanche.benchmarks.classic import CORe50
    from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
        create_generic_benchmark_from_filelists,
    )

    ds = spec["dataset"]
    scenario = ds.get("scenario")
    if scenario is None:
        name = ds["name"]
        scenario = {
            "core50_ni": "ni",
            "core50_nc": "nc",
            "core50_nic": "nicv2_79",  # A14 reconstruction; paper says 79 experiences
        }.get(name)
        if scenario is None:
            raise ValueError(f"unsupported core50 name {name}")
    # Paper input is 32×32. Always mini. ImageNet stats are Avalanche CORe50 defaults
    # (A07/A14 reconstruction), not a paper-stated mean/std. Flip is a function of
    # (aug_seed, image content), not the global RNG.
    aug_seed = int(spec["seeds"]["augmentation"])
    train_tf = SeededCore50Train(aug_seed)
    eval_tf = seeded_core50_eval
    object_level = bool(ds.get("object_level", True))
    run = int(ds.get("run", 0))
    if feedback == "development_val_seen":
        from orion_repro.benchmarks.core50_split import NBATCH, write_core50_dev_split

        split_dir = ds.get("split_dir")
        if split_dir:
            out_dir = Path(split_dir)
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
        else:
            payload = write_core50_dev_split(
                dataset_root=root,
                scenario=scenario,
                run=run,
                object_level=object_level,
            )
            out_dir = Path(payload["out_dir"])
        n_exp = NBATCH[scenario]
        img_root = root / "core50_32x32"
        train_lists = [out_dir / f"train_batch_{k:02d}_filelist.txt" for k in range(n_exp)]
        val_list = out_dir / "val_filelist.txt"
        missing = [p for p in train_lists + [val_list] if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "CORe50 development filelists missing: "
                + ", ".join(str(p) for p in missing[:5])
            )
        bench = create_generic_benchmark_from_filelists(
            img_root,
            train_lists,
            [val_list],
            task_labels=[0 for _ in range(n_exp)],
            complete_test_set_only=True,
            train_transform=train_tf,
            eval_transform=eval_tf,
        )
        setattr(bench, "n_classes", 50 if object_level else 10)
        return bench
    if feedback not in {"official_test_seen"}:
        raise ValueError(f"unhandled core50 feedback_source {feedback}")
    bench = CORe50(
        scenario=scenario,
        run=run,
        object_lvl=object_level,
        mini=True,
        train_transform=train_tf,
        eval_transform=eval_tf,
        dataset_root=root,
    )
    setattr(bench, "n_classes", 50 if object_level else 10)
    return bench



def split_endless_train_indices(
    n: int,
    *,
    experience_id: int,
    seed: int = DEV_SPLIT_SEED,
    val_fraction: float = DEV_VAL_FRACTION,
) -> tuple[list[int], list[int]]:
    """Frame-random development splitting is unsupported; use grouped file paths."""
    raise ValueError("frame-random Endless split retired; use versioned grouped file-path split")


def _build_endless(spec: dict[str, Any], root: Path, feedback: str):
    """Build Endless-Sim classification streams at paper 32×32.

    Do not call Avalanche ``EndlessCLSim``: that wrapper accepts patch_size
    but never forwards it, so ClassificationSubSequence always resizes to 64.
    """
    from avalanche.benchmarks.datasets.endless_cl_sim.endless_cl_sim import (
        EndlessCLSimDataset,
    )
    from avalanche.benchmarks.scenarios.deprecated.generators import (
        dataset_benchmark,
    )
    from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
    from torchvision.transforms import Compose, ToTensor

    ds = spec["dataset"]
    name = ds["name"]
    scenario = ds.get("scenario") or {
        "endless_ic": "Classes",
        "endless_il": "Illumination",
        "endless_wc": "Weather",
    }.get(name)
    if scenario is None:
        raise ValueError(f"unsupported endless name {name}")
    if feedback not in {"development_val_seen", "official_test_seen"}:
        raise ValueError(f"unhandled endless feedback_source {feedback}")
    # Paper Table 2 input is 32×32. Avalanche default patch_size is 64.
    patch_size = int(ds.get("patch_size", 32))
    if patch_size != 32:
        raise ValueError(f"paper Endless input is 32; got patch_size={patch_size}")
    n_exp = int(ds["n_experiences"])
    to_tensor = Compose([ToTensor()])
    raw = EndlessCLSimDataset(
        root=root,
        scenario=scenario,
        patch_size=patch_size,
        transform=None,
        download=False,
        semseg=False,
    )
    if n_exp > len(raw):
        raise ValueError(f"Endless {name} has {len(raw)} subsequences, need {n_exp}")
    train_datasets = []
    eval_datasets = []
    from torch.utils.data import Subset

    # Avalanche uses unsorted glob/scandir; experience identity must be explicit.
    def sequence_id(dataset):
        return int(Path(dataset.file_paths[0]).parent.parent.name)
    raw.train_sub_sequence_datasets.sort(key=sequence_id)
    raw.test_sub_sequence_datasets.sort(key=sequence_id)
    split_records = []
    for i in range(n_exp):
        train_data, official_test = raw[i]
        if sequence_id(train_data) != i or sequence_id(official_test) != i:
            raise ValueError("Endless train/test subsequence identity mismatch")
        # Canonicalize patch order, keeping targets paired.
        for subset in (train_data, official_test):
            pairs = sorted(zip(subset.file_paths, subset.targets), key=lambda item: (Path(item[0]).parent.name, int(Path(item[0]).stem.rsplit("_", 1)[1])))
            subset.file_paths, subset.targets = map(list, zip(*pairs))
        train_data.transform = to_tensor
        official_test.transform = to_tensor
        if feedback == "development_val_seen":
            from orion_repro.benchmarks.endless_split import POLICY, grouped_endless_indices
            if ds.get("split_policy") != POLICY:
                raise ValueError("Endless development requires explicit split_policy=" + POLICY)
            train_idx, val_idx, excluded_idx = grouped_endless_indices(
                train_data.file_paths, experience_id=i,
                seed=int(ds["split_seed"]), val_fraction=float(ds["val_fraction"]),
                embargo=int(ds["embargo_patches"]),
            )
            split_records.append({"experience": i, "policy": POLICY,
                "paths": [str(Path(p).relative_to(root)) for p in train_data.file_paths],
                "train_indices": train_idx, "val_indices": val_idx, "excluded_indices": excluded_idx})
            train_view = Subset(train_data, train_idx)
            eval_view = Subset(train_data, val_idx)
        else:
            train_view = train_data
            eval_view = official_test
        train_datasets.append(
            _make_taskaware_classification_dataset(dataset=train_view, task_labels=0)
        )
        eval_datasets.append(
            _make_taskaware_classification_dataset(dataset=eval_view, task_labels=0)
        )
    bench = dataset_benchmark(train_datasets, eval_datasets)
    setattr(bench, "n_classes", 5)
    setattr(bench, "orion_split_records", split_records)
    return bench


def class_id(label) -> int:
    """Map dataset class tokens to integer ids.

    Endless-Sim stores folder names (BG/Tree/Car/People/Streetlamp) as
    targets. __getitem__ already maps them; Avalanche still exposes the
    strings on classes_in_this_experience.
    """
    if isinstance(label, (int, bool)):
        return int(label)
    try:
        return int(label)
    except (TypeError, ValueError):
        pass
    from avalanche.benchmarks.datasets.endless_cl_sim.endless_cl_sim_data import (
        default_classification_labelmap,
    )
    if label in default_classification_labelmap:
        return int(default_classification_labelmap[label])
    raise ValueError(f"non-integer class label {label!r}")


def experience_class_map(benchmark) -> list[list[int]]:
    mapping = []
    for exp in benchmark.train_stream:
        classes = getattr(exp, "classes_in_this_experience", None)
        if classes is None:
            classes = sorted({class_id(y) for _, y, *rest in exp.dataset})
        mapping.append(sorted({class_id(c) for c in classes}))
    return mapping
