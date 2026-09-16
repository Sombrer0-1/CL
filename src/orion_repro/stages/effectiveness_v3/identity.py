"""Source / environment / platform identity closure for freeze and reuse v4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orion_repro.provenance import sha256_file, snapshot_source_tree
from orion_repro.runner.reuse import actual_training_packages, platform_fingerprint
from orion_repro.stages.effectiveness_v3.constants import STUDY_ID
from orion_repro.stages.effectiveness_v3.util import canonical_hash


def source_identity(root: Path, *, design_path: Path) -> dict[str, Any]:
    snap = snapshot_source_tree(root)
    manifests = {
        "core50_nc_dev": "data/manifests/core50_nc_run0_dev_split_seed17.json",
        "core50_nc_formal": "data/manifests/core50_nc_run0.json",
        "splitcifar100_dev": "data/manifests/cifar100_dev_split_seed17.json",
        "splitcifar100_formal": "data/manifests/paper_feedback/cifar100_official.json",
    }
    data_hashes = {}
    for name, rel in manifests.items():
        path = root / rel
        data_hashes[name] = sha256_file(path) if path.is_file() else None
    lock = root / "requirements.lock.txt"
    return {
        "study_id": STUDY_ID,
        "source_hash": snap["aggregate_sha256"],
        "source_n_files": snap["n_files"],
        "design_hash": sha256_file(design_path) if design_path.is_file() else None,
        "environment_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "environment_actual": actual_training_packages(),
        "platform": platform_fingerprint(),
        "data_hashes": data_hashes,
        "identity_hash": None,
    }


def close_identity(identity: dict[str, Any]) -> dict[str, Any]:
    payload = dict(identity)
    payload.pop("identity_hash", None)
    payload["identity_hash"] = canonical_hash(payload)
    return payload
