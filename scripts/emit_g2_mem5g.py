"""Clone the mem8g pressure-admission probes onto mem5g. kind=probe. Not a freeze.

5G sits between the observed CIFAR100 S0 board peak (~3.97 GiB) and the
pre-registered high-batch peak (~4.74 GiB). 4G would sit under the S0 peak.
6G would still leave about 1 GiB on that high-batch run, the same GiB-scale
slack class already judged unestablished at 8G.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem5g"
SEQ = 1100
CAP = "mem5g"
TASK_IDS: list[str] = []
YAML_NAMES: list[str] = []


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def write_task(task_id: str, argv_extra: list[str]) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": next_seq(),
        "kind": "probe",
        "argv": ["/home/zhuzetong/miniconda3/envs/orion/bin/python", *argv_extra],
        "required_capacity": CAP,
        "batch_id": BATCH,
    }
    dest = TASK / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    TASK_IDS.append(task_id)
    print("task", dest.name)


def emit_yaml_task(src_name: str, *, stem: str, run_id: str, task_id: str, notes: str) -> None:
    text = (CFG / src_name).read_text(encoding="utf-8")
    text = text.replace("mem8g", "mem5g")
    text = re.sub(r"^run_id: .*$", f"run_id: {run_id}", text, count=1, flags=re.M)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("notes:"):
            lines[i] = f"notes: {json.dumps(notes)}"
    dest = CFG / f"{stem}.yaml"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)
    print("yaml", dest.name)
    write_task(task_id, ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{dest.name}"])


def main() -> None:
    TASK.mkdir(parents=True, exist_ok=True)
    write_task("g2-idle-mem5g", ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"])
    emit_yaml_task(
        "g1_cifar100_exp1_mem8g.yaml",
        stem="g1_cifar100_exp1_mem5g",
        run_id="fullmem_v4_g1_cifar100_exp1_mem5g",
        task_id="g1-cifar100-exp1-mem5g",
        notes="G1 mem5g one-experience S0. Admission probe, not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_s0_seed17_mem8g.yaml",
        stem="g2_cifar100_s0_seed17_mem5g",
        run_id="fullmem_v4_g2_cifar100_s0_seed17_mem5g",
        task_id="g2-dev-cifar100-s0-seed17-mem5g",
        notes="G2 mem5g CIFAR100 S0 seed=17 pass 1. Tight needs two complete S0. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_s0_repeat_seed17_mem8g.yaml",
        stem="g2_cifar100_s0_repeat_seed17_mem5g",
        run_id="fullmem_v4_g2_cifar100_s0_repeat_seed17_mem5g",
        task_id="g2-dev-cifar100-s0-repeat-seed17-mem5g",
        notes="G2 mem5g CIFAR100 S0 seed=17 pass 2. Tight needs two complete S0. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_batch256_seed17_mem8g.yaml",
        stem="g2_cifar100_batch256_seed17_mem5g",
        run_id="fullmem_v4_g2_cifar100_batch256_seed17_mem5g",
        task_id="g2-dev-cifar100-batch256-seed17-mem5g",
        notes="G2 mem5g high-batch pressure identifiability. Not Orion success. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_replay2000_seed17_mem8g.yaml",
        stem="g2_cifar100_replay2000_seed17_mem5g",
        run_id="fullmem_v4_g2_cifar100_replay2000_seed17_mem5g",
        task_id="g2-dev-cifar100-replay2000-seed17-mem5g",
        notes="G2 mem5g high-replay pressure identifiability. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_er_gem_ewc_seed17_mem8g.yaml",
        stem="g2_cifar100_er_gem_ewc_seed17_mem5g",
        run_id="fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem5g",
        task_id="g2-dev-cifar100-er-gem-ewc-seed17-mem5g",
        notes="G2 mem5g CIFAR100 ER+GEM+EWC seed=17. Profiling set, not a freeze.",
    )
    emit_yaml_task(
        "g2_core50_nc_s0_seed17_mem8g.yaml",
        stem="g2_core50_nc_s0_seed17_mem5g",
        run_id="fullmem_v4_g2_core50_nc_s0_seed17_mem5g",
        task_id="g2-dev-core50-nc-s0-seed17-mem5g",
        notes="G2 mem5g CORe50-NC S0 seed=17. Not a freeze.",
    )
    write_task("g2-occupancy-mem5g", ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"])
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "note": "5G admission calibration. S0 twice + high-resource static. kind=probe only. Not tight until the runs say so.",
    }
    (TASK / "_batch_g2_dev_mem5g.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(TASK_IDS))


if __name__ == "__main__":
    main()
