"""Clone G2 profiling yamls onto mem32g with new run IDs. Does not overwrite mem64g runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem32g"
SEQ = 400


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def write_task(task_id: str, seq: int, argv_extra: list[str]) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": seq,
        "kind": "probe",
        "argv": ["/home/zhuzetong/miniconda3/envs/orion/bin/python", *argv_extra],
        "required_capacity": "mem32g",
        "batch_id": BATCH,
    }
    dest = TASK / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest.name)


def clone(
    src_name: str,
    *,
    stem: str,
    run_id: str,
    experiment_id: str,
    method_id: str | None = None,
    notes: str,
) -> str:
    text = (CFG / src_name).read_text(encoding="utf-8")
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit(f"no run_id in {src_name}")
    text = text.replace(f"run_id: {old_run}", f"run_id: {run_id}", 1)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {experiment_id}", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {experiment_id}", text, count=1, flags=re.M)
    if method_id:
        text = re.sub(r"^method_id: .*$", f"method_id: {method_id}", text, count=1, flags=re.M)
    text = text.replace("board_capacity_id: mem64g", "board_capacity_id: mem32g")
    text = text.replace(
        "  model: 0\n  stream: 0\n  replay: 0\n  augmentation: 0",
        "  model: 17\n  stream: 17\n  replay: 17\n  augmentation: 17",
        1,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("notes:"):
            lines[i] = f"notes: {json.dumps(notes)}"
    text = "\n".join(lines) + "\n"
    dest = CFG / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    print("yaml", dest.name)
    return stem


def main() -> None:
    TASK.mkdir(parents=True, exist_ok=True)
    items = [
        clone(
            "g1_cifar100_exp1.yaml",
            stem="g1_cifar100_exp1_mem32g",
            run_id="fullmem_v4_g1_cifar100_exp1_mem32g",
            experiment_id="V4_g1_board_mem32g_real_data",
            notes="G1 mem32g one-experience S0. Not a freeze. Does not overwrite mem64g exp1.",
        ),
        clone(
            "g2_cifar100_s0_hash_off_seed17.yaml",
            stem="g2_cifar100_s0_seed17_mem32g",
            run_id="fullmem_v4_g2_cifar100_s0_seed17_mem32g",
            experiment_id="V4_g2_dev_mem32g",
            method_id="S0",
            notes="G2 mem32g CIFAR100 S0 seed=17 profiling. Not a freeze.",
        ),
        clone(
            "g2_cifar100_batch256.yaml",
            stem="g2_cifar100_batch256_seed17_mem32g",
            run_id="fullmem_v4_g2_cifar100_batch256_seed17_mem32g",
            experiment_id="V4_g2_dev_mem32g",
            notes="G2 mem32g high-batch pressure identifiability, seed=17. Not Orion success.",
        ),
        clone(
            "g2_cifar100_replay2000.yaml",
            stem="g2_cifar100_replay2000_seed17_mem32g",
            run_id="fullmem_v4_g2_cifar100_replay2000_seed17_mem32g",
            experiment_id="V4_g2_dev_mem32g",
            notes="G2 mem32g high-replay pressure identifiability, seed=17.",
        ),
        clone(
            "g2_cifar100_er_gem_ewc_seed17.yaml",
            stem="g2_cifar100_er_gem_ewc_seed17_mem32g",
            run_id="fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem32g",
            experiment_id="V4_g2_dev_mem32g",
            notes="G2 mem32g ER+GEM+EWC seed=17 profiling set. Not a freeze.",
        ),
        clone(
            "g2_core50_nc_s0_seed17.yaml",
            stem="g2_core50_nc_s0_seed17_mem32g",
            run_id="fullmem_v4_g2_core50_nc_s0_seed17_mem32g",
            experiment_id="V4_g2_dev_mem32g",
            notes="G2 mem32g CORe50-NC S0 seed=17. Not a freeze.",
        ),
    ]
    mapping = {
        "g1_cifar100_exp1_mem32g": "g1-cifar100-exp1-mem32g",
        "g2_cifar100_s0_seed17_mem32g": "g2-dev-cifar100-s0-seed17-mem32g",
        "g2_cifar100_batch256_seed17_mem32g": "g2-dev-cifar100-batch256-seed17-mem32g",
        "g2_cifar100_replay2000_seed17_mem32g": "g2-dev-cifar100-replay2000-seed17-mem32g",
        "g2_cifar100_er_gem_ewc_seed17_mem32g": "g2-dev-cifar100-er-gem-ewc-seed17-mem32g",
        "g2_core50_nc_s0_seed17_mem32g": "g2-dev-core50-nc-s0-seed17-mem32g",
    }
    for stem in items:
        write_task(
            mapping[stem],
            next_seq(),
            ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
        )
    write_task(
        "g2-occupancy-mem32g",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    )
    write_task(
        "g2-idle-mem32g",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"],
    )


if __name__ == "__main__":
    main()
