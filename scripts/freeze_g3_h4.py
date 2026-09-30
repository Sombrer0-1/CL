"""Write the G3 freeze that admits only formal H4.

One boot capacity, mem=8G, plus the already measured 2.5 GiB H4 hold.
Static three-gear pressure, Endless official protocol, and prefetch net
benefit stay unestablished. Does not upgrade packages or change boot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import atomic_write_json, sha256_file, utc_now

ROOT_PREFIXES = ("src", "tests", "scripts", "docs")
SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache"}
MANIFESTS = (
    "data/manifests/core50_nc_run0_dev_split_seed17.json",
    "data/manifests/core50_nicv2_79_run0_dev_split_seed17.json",
)


def _snapshot(root: Path) -> dict:
    files: list[tuple[str, str]] = []
    for prefix in ROOT_PREFIXES:
        base = root / prefix
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            rel = path.relative_to(root).as_posix()
            files.append((rel, sha256_file(path)))
    files.sort()
    digest = hashlib.sha256()
    for rel, file_hash in files:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"n_files": len(files), "aggregate_sha256": digest.hexdigest(), "files": files}


def main() -> None:
    root = repo_root()
    out = root / "reports" / STUDY_ID / "g3"
    out.mkdir(parents=True, exist_ok=True)
    snap = _snapshot(root)
    (out / "source_files_g4.txt").write_text(
        "".join(f"{file_hash}  {rel}\n" for rel, file_hash in snap["files"]),
        encoding="utf-8",
        newline="\n",
    )
    pip = subprocess.check_output(
        ["/home/zhuzetong/miniconda3/envs/orion/bin/python", "-m", "pip", "freeze"],
        text=True,
    )
    lock_path = out / "pip_freeze_g4.txt"
    lock_path.write_text(pip if pip.endswith("\n") else pip + "\n", encoding="utf-8", newline="\n")
    identity = collect_identity(include_cuda=True)
    manifests = {rel: sha256_file(root / rel) for rel in MANIFESTS}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    cap = (identity.get("capacity") or {}).get("capacity_id")
    if cap != "mem8g":
        raise SystemExit(f"refuse freeze: capacity={cap}")
    payload = {
        "study_id": STUDY_ID,
        "design_version": "design-v1",
        "frozen_at_utc": utc_now(),
        "admits_kind_train": True,
        "admitted_batches": ["g4-h4-mem8g"],
        "capacity_id": "mem8g",
        "not_tight": True,
        "static_pressure": "unestablished_at_mem64g_mem32g_mem16g_mem8g",
        "h4_hold_bytes": 2684354560,
        "h4_sequences": ["low_high_low", "high_low_high"],
        "dynamic_sstar": {
            "core50_nc": {"low_high_low": {"new_batch": 64, "replay_capacity": 2000}, "high_low_high": {"new_batch": 16, "replay_capacity": 2000}},
            "core50_nic": {"low_high_low": {"new_batch": 16, "replay_capacity": 2000}, "high_low_high": {"new_batch": 16, "replay_capacity": 2000}},
        },
        "formal_seeds": [0, 1, 2],
        "data_split_seed": 17,
        "source_snapshot_sha256": snap["aggregate_sha256"],
        "source_snapshot_n_files": snap["n_files"],
        "environment_lock_sha256": sha256_file(lock_path),
        "dataset_manifest_sha256": manifests,
        "git_head_not_the_snapshot": head,
        "boot_id": identity.get("boot_id"),
        "mem_total_kb": (identity.get("meminfo_kb") or {}).get("MemTotal"),
        "unestablished": [
            "static_tight_mid_wide",
            "h1_h2_h3_h5_h6_h7_h8_formal",
            "endless_official_protocol",
            "prefetch_net_benefit",
            "unrecoverable_board_crash",
            "cifar100_64g_sstar_is_not_h4_sstar",
        ],
    }
    atomic_write_json(out / "FREEZE.json", payload)
    print(json.dumps({k: payload[k] for k in ("source_snapshot_sha256", "environment_lock_sha256", "capacity_id", "boot_id", "admitted_batches")}, indent=2))


if __name__ == "__main__":
    main()
