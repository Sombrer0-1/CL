"""G2 data inventory and public-dataset fetch. Does not rewrite git-tracked manifests."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.queue import ensure_dirs
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now


def _dir_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    files = [p for p in path.rglob("*") if p.is_file()]
    nbytes = sum(p.stat().st_size for p in files)
    return {
        "exists": True,
        "path": str(path),
        "n_files": len(files),
        "bytes": nbytes,
    }


def inventory(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    payload = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "note": "Git-tracked data/manifests are historical identity; this inventory does not rewrite them.",
        "streams": {
            "splitcifar10": {
                "paper_experiences": 10,
                "raw": _dir_info(root / "data" / "raw" / "cifar10"),
                "dev_split": (root / "data" / "manifests" / "cifar10_dev_split_seed17.json").is_file(),
                "class_order_10exp": [4, 1, 7, 5, 3, 9, 0, 8, 6, 2],
                "ready": False,
            },
            "splitcifar100": {
                "paper_experiences": 10,
                "raw": _dir_info(root / "data" / "raw" / "cifar100"),
                "dev_split": (root / "data" / "manifests" / "cifar100_dev_split_seed17.json").is_file(),
                "ready": False,
            },
            "core50_nc": {
                "paper_experiences": 9,
                "raw": _dir_info(root / "data" / "raw" / "core50"),
                "dev_split": (root / "data" / "processed" / "core50_nc_run0_dev_split_seed17").is_dir(),
                "ready": False,
            },
            "core50_ni": {
                "paper_experiences": 8,
                "raw": _dir_info(root / "data" / "raw" / "core50"),
                "dev_split_manifest": (root / "data" / "manifests" / "core50_ni_run0_dev_split_seed17.json").is_file(),
                "ready": False,
            },
            "core50_nic": {
                "paper_experiences": 79,
                "raw": _dir_info(root / "data" / "raw" / "core50"),
                "dev_split_manifest": (root / "data" / "manifests" / "core50_nicv2_79_run0_dev_split_seed17.json").is_file(),
                "nic_mapping": "Avalanche nicv2_79 reconstruction (A14); not author-confirmed SKU",
                "ready": False,
            },
            "endless_ic": {"paper_experiences": 4, "raw": _dir_info(root / "data" / "raw" / "endless_cl_sim"), "ready": False},
            "endless_il": {"paper_experiences": 5, "raw": _dir_info(root / "data" / "raw" / "endless_cl_sim"), "ready": False},
            "endless_wc": {"paper_experiences": 5, "raw": _dir_info(root / "data" / "raw" / "endless_cl_sim"), "ready": False},
        },
    }
    cifar10 = payload["streams"]["splitcifar10"]
    cifar10["ready"] = bool(cifar10["raw"].get("bytes", 0) > 10_000_000 and cifar10["dev_split"])
    cifar100 = payload["streams"]["splitcifar100"]
    cifar100["ready"] = bool(cifar100["raw"].get("bytes", 0) > 10_000_000 and cifar100["dev_split"])
    payload["streams"]["core50_nc"]["ready"] = bool(
        payload["streams"]["core50_nc"]["raw"].get("bytes", 0) > 1_000_000_000 and payload["streams"]["core50_nc"]["dev_split"]
    )
    payload["streams"]["core50_ni"]["ready"] = bool(
        payload["streams"]["core50_ni"]["raw"].get("bytes", 0) > 1_000_000_000
        and payload["streams"]["core50_ni"]["dev_split_manifest"]
    )
    payload["streams"]["core50_nic"]["ready"] = bool(
        payload["streams"]["core50_nic"]["raw"].get("bytes", 0) > 1_000_000_000
        and payload["streams"]["core50_nic"]["dev_split_manifest"]
    )
    endless_root = root / "data" / "raw" / "endless_cl_sim"
    names = {
        "endless_ic": "IncrementalClasses_Classification.zip",
        "endless_il": "IncrementalLighting_Classification.zip",
        "endless_wc": "IncrementalWeather_Classification.zip",
    }
    for key, filename in names.items():
        z = endless_root / filename
        payload["streams"][key]["zip_exists"] = z.is_file()
        payload["streams"][key]["ready"] = z.is_file() and z.stat().st_size > 100_000_000
    return payload


def save_inventory(root: Path | None = None) -> dict[str, Any]:
    root = (root or repo_root()).resolve()
    payload = inventory(root)
    dest = assert_write_path(root / "reports" / STUDY_ID / "data" / "inventory.json", root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    payload["output"] = str(dest)
    return payload


def prepare_cifar10_without_manifest_rewrite() -> dict[str, Any]:
    from orion_repro.prepare_data import prepare_cifar10

    root = repo_root()
    payload = prepare_cifar10(root / "data" / "raw" / "cifar10")
    dest = assert_write_path(root / "reports" / STUDY_ID / "data" / "cifar10_prepare.json", root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, payload)
    return {"output": str(dest), "n_total": payload.get("n_total"), "rewrote_git_manifest": False}


def prepare_endless_without_manifest_rewrite(scenario: str = "all") -> dict[str, Any]:
    from orion_repro.prepare_endless import SCENARIOS, download_scenario, extract_scenario

    from orion_repro.stages.fullmem_v4.util import StageError

    root = repo_root()
    dest_dir = root / "data" / "raw" / "endless_cl_sim"
    free = shutil.disk_usage(root).free
    if free < 20 * 1024 * 1024 * 1024:
        raise StageError(f"disk too low for Endless download/extract: {free} bytes")
    names = list(SCENARIOS) if scenario == "all" else [scenario]
    rows = []
    for name in names:
        downloaded = download_scenario(name, dest_dir)
        extracted = extract_scenario(name, dest_dir)
        rows.append({"download": downloaded, "extract": extracted})
    payload = {
        "study_id": STUDY_ID,
        "source": "zenodo.4899267 classification zips; not video",
        "prepared_at_utc": utc_now(),
        "rewrote_git_manifest": False,
        "historical_manifest_kept": "data/manifests/endless_cl_sim.json",
        "a22_not_official": True,
        "scenarios": rows,
    }
    out = assert_write_path(root / "reports" / STUDY_ID / "data" / "endless_download.json", root)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    payload["output"] = str(out)
    return payload


def prepare_core50_dev_splits_without_manifest_rewrite() -> dict[str, Any]:
    """Write NI/NIC development filelists under reports/fullmem_v4/. Do not rewrite git manifests."""
    from orion_repro.benchmarks.core50_split import write_core50_dev_split

    root = repo_root()
    dataset_root = root / "data" / "raw" / "core50"
    rows = []
    for scenario in ("ni", "nicv2_79"):
        out_dir = assert_write_path(
            root / "reports" / STUDY_ID / "data" / f"core50_{scenario}_run0_dev_split_seed17",
            root,
        )
        payload = write_core50_dev_split(
            dataset_root=dataset_root,
            scenario=scenario,
            run=0,
            object_level=True,
            seed=17,
            val_fraction=0.20,
            out_dir=out_dir,
        )
        rows.append(
            {
                "scenario": scenario,
                "out_dir": payload.get("out_dir"),
                "n_experiences": payload.get("n_experiences"),
                "n_dev_train": payload.get("n_dev_train"),
                "n_dev_val": payload.get("n_dev_val"),
            }
        )
    summary = {
        "study_id": STUDY_ID,
        "prepared_at_utc": utc_now(),
        "rewrote_git_manifest": False,
        "historical_manifests_kept": [
            "data/manifests/core50_ni_run0_dev_split_seed17.json",
            "data/manifests/core50_nicv2_79_run0_dev_split_seed17.json",
        ],
        "streams": rows,
    }
    dest = assert_write_path(root / "reports" / STUDY_ID / "data" / "core50_dev_splits.json", root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, summary)
    summary["output"] = str(dest)
    return summary
