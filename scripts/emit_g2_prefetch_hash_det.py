"""Emit seed=17 S0 prefetch off/on probes with deterministic_algorithms.

kind=probe only. Does not freeze. Does not claim H7 benefit until param hashes match.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFF_SRC = ROOT / "configs" / "fullmem_v4" / "g2_cifar100_s0_hash_off_seed17.yaml"
ON_SRC = ROOT / "configs" / "fullmem_v4" / "g2_cifar100_s0_hash_on_seed17.yaml"
BATCH = "g2-dev-prefetch-hash-det"
SEQ = 940


def _patch(text: str, *, run_id: str, method_id: str, notes: str) -> str:
    text = text.replace("run_id: fullmem_v4_g2_cifar100_s0_hash_off_seed17", f"run_id: {run_id}", 1)
    text = text.replace("run_id: fullmem_v4_g2_cifar100_s0_hash_on_seed17", f"run_id: {run_id}", 1)
    text = text.replace("method_id: S0_hash_off", f"method_id: {method_id}", 1)
    text = text.replace("method_id: S0_hash_on", f"method_id: {method_id}", 1)
    if "deterministic_algorithms:" not in text:
        text = text.replace(
            "  allow_tf32: false\n  cudnn_benchmark: false\n",
            "  allow_tf32: false\n  cudnn_benchmark: false\n  deterministic_algorithms: true\n",
            1,
        )
    text = text.replace("pressure_role: g2_dev_calibration_mem64g", "pressure_role: g2_dev_calibration_mem16g", 1)
    text = text.replace("board_capacity_id: mem64g", "board_capacity_id: mem16g", 1)
    text = text.replace(
        'notes: "G2 H7 supply/param hash: S0 prefetch off, record_hashes on, seed=17. Not a freeze."',
        f"notes: {json.dumps(notes)}",
        1,
    )
    text = text.replace(
        'notes: "G2 H7 supply/param hash: S0 prefetch on queue_depth=2, record_hashes on, seed=17. Not a freeze."',
        f"notes: {json.dumps(notes)}",
        1,
    )
    return text


def write_task(task_id: str, seq: int, config_rel: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": seq,
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.run",
            "--config",
            config_rel,
        ],
        "required_capacity": "mem16g",
        "batch_id": BATCH,
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest.name)


def main() -> None:
    off_text = _patch(
        OFF_SRC.read_text(encoding="utf-8"),
        run_id="fullmem_v4_g2_cifar100_s0_hash_off_det_seed17_mem16g",
        method_id="S0_hash_off_det",
        notes="G2 H7 diagnostic: S0 prefetch off, record_hashes, deterministic_algorithms, seed=17, mem16g. Not a freeze.",
    )
    on_text = _patch(
        ON_SRC.read_text(encoding="utf-8"),
        run_id="fullmem_v4_g2_cifar100_s0_hash_on_det_seed17_mem16g",
        method_id="S0_hash_on_det",
        notes="G2 H7 diagnostic: S0 prefetch on queue_depth=2, record_hashes, deterministic_algorithms, seed=17, mem16g. Not a freeze.",
    )
    off_name = "g2_cifar100_s0_hash_off_det_seed17_mem16g.yaml"
    on_name = "g2_cifar100_s0_hash_on_det_seed17_mem16g.yaml"
    (ROOT / "configs" / "fullmem_v4" / off_name).write_text(off_text, encoding="utf-8", newline="\n")
    (ROOT / "configs" / "fullmem_v4" / on_name).write_text(on_text, encoding="utf-8", newline="\n")
    print("yaml", off_name)
    print("yaml", on_name)
    write_task("g2-dev-cifar100-s0-hash-off-det-seed17-mem16g", SEQ, f"configs/fullmem_v4/{off_name}")
    write_task("g2-dev-cifar100-s0-hash-on-det-seed17-mem16g", SEQ + 1, f"configs/fullmem_v4/{on_name}")
    write_task_occ = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": "g2-occupancy-h7-det",
        "seq": SEQ + 2,
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.stages.fullmem_v4",
            "g2-occupancy",
        ],
        "required_capacity": "mem16g",
        "batch_id": BATCH,
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / "g2-occupancy-h7-det.json"
    dest.write_text(json.dumps(write_task_occ, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest.name)
    batch = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": "mem16g",
        "tasks": [
            "g2-dev-cifar100-s0-hash-off-det-seed17-mem16g",
            "g2-dev-cifar100-s0-hash-on-det-seed17-mem16g",
            "g2-occupancy-h7-det",
        ],
        "yamls": [off_name, on_name],
    }
    man = ROOT / "experiments" / "fullmem_v4" / "tasks" / "_batch_g2_dev_prefetch_hash_det.json"
    man.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("manifest", man.name)


if __name__ == "__main__":
    main()
