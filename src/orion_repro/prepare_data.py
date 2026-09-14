"""Download public datasets and write manifests (T02)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_cifar10(root: Path) -> dict:
    from torchvision.datasets import CIFAR10

    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(root=str(root), train=True, download=True)
    test = CIFAR10(root=str(root), train=False, download=True)
    tar = root / "cifar-10-python.tar.gz"
    train_labels = [int(y) for y in train.targets]
    test_labels = [int(y) for y in test.targets]
    payload = {
        "dataset": "cifar10",
        "source": "torchvision.datasets.CIFAR10",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "n_train": len(train),
        "n_test": len(test),
        "n_total": len(train) + len(test),
        "image_size": [32, 32],
        "n_classes": 10,
        "class_names": list(train.classes),
        "train_per_class": dict(sorted(Counter(train_labels).items())),
        "test_per_class": dict(sorted(Counter(test_labels).items())),
        "tarball_sha256": sha256_file(tar) if tar.exists() else None,
        "note": "Official 50k/10k split. Paper Table 2 60,000 is train+test. "
        "Development train/val is a further stratified split of the 50k train set; "
        "see data/manifests/cifar10_dev_split_seed17.json.",
    }
    return payload


def prepare_cifar100(root: Path) -> dict:
    from torchvision.datasets import CIFAR100

    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR100(root=str(root), train=True, download=True)
    test = CIFAR100(root=str(root), train=False, download=True)
    tar = root / "cifar-100-python.tar.gz"
    train_labels = [int(y) for y in train.targets]
    test_labels = [int(y) for y in test.targets]
    return {
        "dataset": "cifar100",
        "source": "torchvision.datasets.CIFAR100",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "n_train": len(train),
        "n_test": len(test),
        "n_total": len(train) + len(test),
        "image_size": [32, 32],
        "n_classes": 100,
        "class_names": list(train.classes),
        "train_per_class": dict(sorted(Counter(train_labels).items())),
        "test_per_class": dict(sorted(Counter(test_labels).items())),
        "tarball_sha256": sha256_file(tar) if tar.exists() else None,
        "note": "Official 50k/10k split. Paper Table 2 60,000 is train+test. SplitCIFAR100 uses 10 experiences, not 100.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cifar10",
            "cifar100",
            "cifar10_dev",
            "cifar100_dev",
            "core50_dev_ni",
            "core50_dev_nc",
            "core50_dev_nic",
        ],
    )
    args = parser.parse_args(argv)
    raw = ROOT / "data" / "raw"
    manifests = ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cifar10":
        payload = prepare_cifar10(raw / "cifar10")
        out = manifests / "cifar10.json"
        expected = 60000
    elif args.dataset == "cifar100":
        payload = prepare_cifar100(raw / "cifar100")
        out = manifests / "cifar100.json"
        expected = 60000
    elif args.dataset in {"cifar10_dev", "cifar100_dev"}:
        from orion_repro.benchmarks.dev_split import write_cifar_dev_manifest

        name = "cifar10" if args.dataset == "cifar10_dev" else "cifar100"
        out = manifests / f"{name}_dev_split_seed17.json"
        payload = write_cifar_dev_manifest(name, raw / name, out)
        print(json.dumps({"wrote": str(out), "n_dev_train": payload["n_dev_train"], "n_dev_val": payload["n_dev_val"]}, indent=2))
        return
    elif args.dataset.startswith("core50_dev_"):
        from orion_repro.benchmarks.core50_split import write_core50_dev_split

        scenario = {"core50_dev_ni": "ni", "core50_dev_nc": "nc", "core50_dev_nic": "nicv2_79"}[
            args.dataset
        ]
        payload = write_core50_dev_split(
            dataset_root=raw / "core50",
            scenario=scenario,
            run=0,
            object_level=True,
        )
        out = manifests / f"core50_{scenario}_run0_dev_split_seed17.json"
        slim = {k: v for k, v in payload.items() if k != "assignments"}
        out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(json.dumps({"wrote": str(out), "out_dir": payload["out_dir"], "n_dev_train": payload["n_dev_train"], "n_dev_val": payload["n_dev_val"]}, indent=2))
        return
    else:
        raise SystemExit(f"unsupported {args.dataset}")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_total": payload["n_total"]}, indent=2))
    if payload["n_total"] != expected:
        raise SystemExit(f"unexpected total {payload['n_total']}")


if __name__ == "__main__":
    main()
