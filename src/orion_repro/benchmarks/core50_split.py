"""CORe50 development split by sequence directory, not random frames (PLAN §4.3)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion_repro.rng import mix_seed

ROOT = Path(__file__).resolve().parents[3]

NBATCH = {
    "ni": 8,
    "nc": 9,
    "nicv2_79": 79,
}

SCEN_DIRS = {
    "ni": "batches_filelists/NI_inc",
    "nc": "batches_filelists/NC_inc",
    "nicv2_79": "NIC_v2_79",
}


def _filelist_dir(dataset_root: Path, scenario: str, run: int, object_level: bool) -> Path:
    suffix = "" if object_level else "_cat"
    return dataset_root / (SCEN_DIRS[scenario] + suffix) / f"run{run}"


def _parse_line(line: str) -> tuple[str, str, int] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    path, label_s = raw.split()
    parts = path.split("/")
    if len(parts) < 2:
        seq = parts[0]
    else:
        seq = "/".join(parts[:2])
    return path, seq, int(label_s)


def split_filelist_lines(
    lines: list[str],
    *,
    seed: int,
    val_fraction: float,
    experience_id: int,
) -> tuple[list[str], list[str], dict[str, str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        parsed = _parse_line(line)
        if parsed is None:
            continue
        _path, seq, _y = parsed
        groups[seq].append(line if line.endswith("\n") else line + "\n")
    assignment: dict[str, str] = {}
    train_lines: list[str] = []
    val_lines: list[str] = []
    seqs = sorted(groups)
    for seq in seqs:
        u = mix_seed(seed, experience_id, seq, "core50_seq_v1") / float(2**63)
        role = "val" if u < val_fraction else "train"
        assignment[seq] = role
    if all(assignment[s] == "val" for s in seqs) and seqs:
        assignment[seqs[0]] = "train"
    if all(assignment[s] == "train" for s in seqs) and len(seqs) > 1:
        assignment[seqs[-1]] = "val"
    for seq in seqs:
        bucket = val_lines if assignment[seq] == "val" else train_lines
        bucket.extend(groups[seq])
    return train_lines, val_lines, assignment


def write_core50_dev_split(
    *,
    dataset_root: Path,
    scenario: str,
    run: int = 0,
    object_level: bool = True,
    seed: int = 17,
    val_fraction: float = 0.20,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    n_exp = NBATCH[scenario]
    src = _filelist_dir(dataset_root, scenario, run, object_level)
    if out_dir is None:
        out_dir = (
            ROOT
            / "data"
            / "processed"
            / f"core50_{scenario}_run{run}_dev_split_seed{seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    all_val: list[str] = []
    train_sizes = []
    val_sizes = []
    assignments = []
    for k in range(n_exp):
        src_file = src / f"train_batch_{k:02d}_filelist.txt"
        lines = src_file.read_text(encoding="utf-8").splitlines(True)
        train_lines, val_lines, assignment = split_filelist_lines(
            lines, seed=seed, val_fraction=val_fraction, experience_id=k
        )
        dest = out_dir / f"train_batch_{k:02d}_filelist.txt"
        dest.write_text("".join(train_lines), encoding="utf-8")
        all_val.extend(val_lines)
        train_sizes.append(len(train_lines))
        val_sizes.append(len(val_lines))
        assignments.append(assignment)
    val_path = out_dir / "val_filelist.txt"
    val_path.write_text("".join(all_val), encoding="utf-8")
    payload = {
        "dataset": f"core50_{scenario}",
        "source": "official_train_filelists_sequence_groups",
        "scenario": scenario,
        "run": run,
        "object_level": object_level,
        "split_seed": seed,
        "val_fraction": val_fraction,
        "sequence_key": "session/object directory",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "src_filelist_dir": str(src),
        "out_dir": str(out_dir),
        "n_experiences": n_exp,
        "train_sizes": train_sizes,
        "val_sizes_per_experience": val_sizes,
        "n_dev_train": int(sum(train_sizes)),
        "n_dev_val": len(all_val),
        "n_sequences": [len(a) for a in assignments],
        "note": (
            "Whole s*/o* sequences stay on one side of the split. "
            "Official test_filelist is unused. Frames of one sequence are not "
            "randomly mixed between train and val."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
