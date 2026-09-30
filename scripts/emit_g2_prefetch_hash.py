"""Emit seed=17 S0 prefetch off/on probes with record_hashes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs" / "fullmem_v4" / "g2_cifar100_s0.yaml"


def write_task(task_id: str, seq: int, argv_extra: list[str], batch_id: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": seq,
        "kind": "probe",
        "argv": ["/home/zhuzetong/miniconda3/envs/orion/bin/python", *argv_extra],
        "required_capacity": "mem64g",
        "batch_id": batch_id,
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest.name)


def variant(*, stem: str, run_id: str, experiment_id: str, method_id: str, prefetch_on: bool, notes: str) -> str:
    text = SRC.read_text(encoding="utf-8")
    text = text.replace("run_id: fullmem_v4_g2_cifar100_s0", f"run_id: {run_id}", 1)
    text = text.replace("- V4_g2_dev_cifar100_s0", f"- {experiment_id}", 1)
    text = text.replace("experiment_id: V4_g2_dev_cifar100_s0", f"experiment_id: {experiment_id}", 1)
    text = text.replace("method_id: S0", f"method_id: {method_id}", 1)
    text = text.replace("  model: 0\n  stream: 0\n  replay: 0\n  augmentation: 0", "  model: 17\n  stream: 17\n  replay: 17\n  augmentation: 17", 1)
    if prefetch_on:
        text = text.replace("prefetch:\n  enabled: false\n  queue_depth: 1", "prefetch:\n  enabled: true\n  queue_depth: 2", 1)
    text = text.replace(
        'notes: "G2 full-stream S0 development calibration on mem64g; observed_only; not a freeze."',
        f'notes: {json.dumps(notes)}',
        1,
    )
    extra = (
        "data_supply:\n"
        "  record_hashes: true\n"
        "  profile: natural_ondemand\n"
        "  profile_path: null\n"
    )
    text = text.replace("measurement:\n", extra + "measurement:\n", 1)
    dest = ROOT / "configs" / "fullmem_v4" / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    print("yaml", dest.name)
    return stem


def main() -> None:
    off = variant(
        stem="g2_cifar100_s0_hash_off_seed17",
        run_id="fullmem_v4_g2_cifar100_s0_hash_off_seed17",
        experiment_id="V4_g2_dev_cifar100_prefetch_hash",
        method_id="S0_hash_off",
        prefetch_on=False,
        notes="G2 H7 supply/param hash: S0 prefetch off, record_hashes on, seed=17. Not a freeze.",
    )
    on = variant(
        stem="g2_cifar100_s0_hash_on_seed17",
        run_id="fullmem_v4_g2_cifar100_s0_hash_on_seed17",
        experiment_id="V4_g2_dev_cifar100_prefetch_hash",
        method_id="S0_hash_on",
        prefetch_on=True,
        notes="G2 H7 supply/param hash: S0 prefetch on queue_depth=2, record_hashes on, seed=17. Not a freeze.",
    )
    write_task(
        "g2-dev-cifar100-s0-hash-off-seed17",
        270,
        ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{off}.yaml"],
        "g2-dev-prefetch-hash",
    )
    write_task(
        "g2-dev-cifar100-s0-hash-on-seed17",
        271,
        ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{on}.yaml"],
        "g2-dev-prefetch-hash",
    )
    write_task("g2-occupancy-d", 272, ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"], "g2-dev-prefetch-hash")
    write_task("g2-cost-b", 273, ["-m", "orion_repro.stages.fullmem_v4", "g2-cost"], "g2-dev-prefetch-hash")


if __name__ == "__main__":
    main()
