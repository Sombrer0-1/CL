"""Write a G3 freeze-pack draft. Does not freeze. Does not admit kind=train."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from orion_repro.provenance import snapshot_source_tree
from orion_repro.stages.fullmem_v4.constants import STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.g2_cost import cost_estimate
from orion_repro.stages.fullmem_v4.g2_inventory import save_inventory
from orion_repro.stages.fullmem_v4.g2_occupancy import occupancy_report
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, atomic_write_text, utc_now


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def main() -> int:
    root = repo_root()
    dest_dir = assert_write_path(root / "reports" / STUDY_ID / "g3")
    dest_dir.mkdir(parents=True, exist_ok=True)

    identity = collect_identity(include_cuda=True)
    atomic_write_json(dest_dir / "identity_live.json", identity)

    source = snapshot_source_tree(root, exclude_generated_v3=True)
    atomic_write_json(
        dest_dir / "source_snapshot.json",
        {
            "kind": source["kind"],
            "created_utc": source["created_utc"],
            "aggregate_sha256": source["aggregate_sha256"],
            "n_files": source["n_files"],
            "git_head_is_not_this_snapshot": True,
            "note": "Uncommitted staging overlay. Git HEAD is still v3. Not a freeze.",
        },
    )
    listing_path = dest_dir / "source_tree_files.jsonl"
    with listing_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in source["files"]:
            handle.write(json.dumps(row) + "\n")

    manifests = {}
    man_dir = root / "data" / "manifests"
    if man_dir.is_dir():
        for path in sorted(man_dir.glob("*.json")):
            manifests[str(path.relative_to(root)).replace("\\", "/")] = _sha256_file(path)

    pip = _run(["/home/zhuzetong/miniconda3/envs/orion/bin/python", "-m", "pip", "freeze"])
    atomic_write_text(dest_dir / "pip_freeze.txt", str(pip.get("stdout") or ""))
    conda = _run(["/home/zhuzetong/miniconda3/envs/orion/bin/conda", "list", "--export"])
    atomic_write_text(dest_dir / "conda_list_export.txt", str(conda.get("stdout") or conda.get("stderr") or ""))

    git_head = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_sb = _run(["git", "-C", str(root), "status", "-sb"])
    git_count = _run(["git", "-C", str(root), "status", "--porcelain"])
    dirty_n = len([line for line in str(git_count.get("stdout") or "").splitlines() if line.strip()])

    nv = _run(["cat", "/etc/nv_tegra_release"])
    uname = _run(["uname", "-a"])
    nvp = _run(["nvpmodel", "-q"])

    occupancy = occupancy_report()
    inventory = save_inventory()
    cost = cost_estimate()

    locks = {
        "requirements.lock.txt": _sha256_file(root / "requirements.lock.txt"),
        "environment.lock.yml": _sha256_file(root / "environment.lock.yml"),
        "pyproject.toml": _sha256_file(root / "pyproject.toml"),
    }

    draft = {
        "study_id": STUDY_ID,
        "design_version": "design-v1",
        "collected_at_utc": utc_now(),
        "not_a_freeze": True,
        "kind_train_still_refused": True,
        "items": {
            "design_revision": {"status": "present", "value": "design-v1"},
            "git_commit": {
                "status": "not_frozen",
                "head": (git_head.get("stdout") or "").strip(),
                "status_sb": (git_sb.get("stdout") or "").splitlines()[:3],
                "dirty_paths_n": dirty_n,
                "note": "HEAD is historical v3. v4 overlay is uncommitted. Cannot bind freeze to this HEAD.",
            },
            "source_tree_snapshot": {
                "status": "draft_only",
                "aggregate_sha256": source["aggregate_sha256"],
                "n_files": source["n_files"],
            },
            "orion_lock": {"status": "missing_or_unfrozen", "hashes": locks, "pip_freeze": str(dest_dir / "pip_freeze.txt")},
            "driver_kernel_cuda": {
                "status": "recorded_live_not_frozen",
                "uname": (uname.get("stdout") or "").strip(),
                "nv_tegra_release": (nv.get("stdout") or "").strip()[:500],
                "nvpmodel": (nvp.get("stdout") or "").strip(),
                "cuda": identity.get("cuda"),
                "boot_id": identity.get("boot_id"),
                "capacity": identity.get("capacity"),
            },
            "data_manifests": {"status": "hashed_not_frozen", "sha256": manifests},
            "actual_budget": {
                "status": "unestablished",
                "pressure_scene": occupancy.get("pressure_scene"),
                "min_available_bytes": occupancy.get("min_available_bytes_across_completed"),
                "note": "16G admitted, not frozen. Do not write a tight/mid/wide freeze from slack.",
            },
            "matrix": {
                "status": "design_denominator_only",
                "design_slots_before_dedup": 1320,
                "cost_output": cost.get("output"),
                "inventory_output": inventory.get("output"),
            },
            "g1_fault_probes": {
                "status": "probes_done_except_hard_oom",
                "done": [
                    "duplicate_submit",
                    "ssh_disconnect",
                    "agent_exit",
                    "worker_exception",
                    "resource_failure_follow_on",
                    "linger_terminate_user",
                    "disk_full_tmpfs",
                    "in_run_reboot",
                ],
                "hard_board_oom": "unconfirmed_by_design",
            },
            "control_machine_backup": {"status": "missing"},
            "h7_param_hash": {"status": "unestablished"},
            "endless_official_protocol": {"status": "unestablished"},
        },
        "outputs": {
            "identity": str(dest_dir / "identity_live.json"),
            "source_snapshot": str(dest_dir / "source_snapshot.json"),
            "source_tree_files": str(listing_path),
        },
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
    }
    atomic_write_json(dest_dir / "freeze_pack_draft.json", draft)
    print(json.dumps({"output": str(dest_dir / "freeze_pack_draft.json"), "not_a_freeze": True, "source": source["aggregate_sha256"], "git_head": draft["items"]["git_commit"]["head"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
