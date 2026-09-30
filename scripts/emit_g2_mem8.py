"""Clone G2 pressure-admission yamls onto mem8g. kind=probe. Not a freeze."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem8g"
SEQ = 1000
CAP = "mem8g"
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


def emit_yaml_task(
    src_name: str,
    *,
    stem: str,
    run_id: str,
    task_id: str,
    notes: str,
    experiment_id: str = "V4_g2_dev_mem8g",
    claim_id: str | None = None,
) -> None:
    text = (CFG / src_name).read_text(encoding="utf-8")
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit(f"no run_id in {src_name}")
    claim = claim_id or experiment_id
    text = text.replace(f"run_id: {old_run}", f"run_id: {run_id}", 1)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {experiment_id}", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {claim}", text, count=1, flags=re.M)
    for old in ("mem64g", "mem32g", "mem16g"):
        text = text.replace(f"board_capacity_id: {old}", f"board_capacity_id: {CAP}")
        text = text.replace(f"pressure_role: g2_dev_calibration_{old}", "pressure_role: g2_dev_calibration_mem8g")
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
    write_task(
        "g2-idle-mem8g",
        ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"],
    )
    emit_yaml_task(
        "g1_cifar100_exp1_mem16g.yaml",
        stem="g1_cifar100_exp1_mem8g",
        run_id="fullmem_v4_g1_cifar100_exp1_mem8g",
        task_id="g1-cifar100-exp1-mem8g",
        notes="G1 mem8g one-experience S0. Admission probe, not a freeze.",
        experiment_id="V4_g1_board_mem8g_real_data",
        claim_id="V4_g1_board_mem8g_real_data",
    )
    emit_yaml_task(
        "g2_cifar100_s0_seed17_mem16g.yaml",
        stem="g2_cifar100_s0_seed17_mem8g",
        run_id="fullmem_v4_g2_cifar100_s0_seed17_mem8g",
        task_id="g2-dev-cifar100-s0-seed17-mem8g",
        notes="G2 mem8g CIFAR100 S0 seed=17 pass 1. Tight needs two complete S0. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_s0_seed17_mem16g.yaml",
        stem="g2_cifar100_s0_repeat_seed17_mem8g",
        run_id="fullmem_v4_g2_cifar100_s0_repeat_seed17_mem8g",
        task_id="g2-dev-cifar100-s0-repeat-seed17-mem8g",
        notes="G2 mem8g CIFAR100 S0 seed=17 pass 2. Tight needs two complete S0. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_batch256_seed17_mem16g.yaml",
        stem="g2_cifar100_batch256_seed17_mem8g",
        run_id="fullmem_v4_g2_cifar100_batch256_seed17_mem8g",
        task_id="g2-dev-cifar100-batch256-seed17-mem8g",
        notes="G2 mem8g high-batch pressure identifiability. Not Orion success. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_replay2000_seed17_mem16g.yaml",
        stem="g2_cifar100_replay2000_seed17_mem8g",
        run_id="fullmem_v4_g2_cifar100_replay2000_seed17_mem8g",
        task_id="g2-dev-cifar100-replay2000-seed17-mem8g",
        notes="G2 mem8g high-replay pressure identifiability. Not a freeze.",
    )
    emit_yaml_task(
        "g2_cifar100_er_gem_ewc_seed17_mem16g.yaml",
        stem="g2_cifar100_er_gem_ewc_seed17_mem8g",
        run_id="fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem8g",
        task_id="g2-dev-cifar100-er-gem-ewc-seed17-mem8g",
        notes="G2 mem8g CIFAR100 ER+GEM+EWC seed=17. Profiling set, not a freeze.",
    )
    emit_yaml_task(
        "g2_core50_nc_s0_seed17_mem16g.yaml",
        stem="g2_core50_nc_s0_seed17_mem8g",
        run_id="fullmem_v4_g2_core50_nc_s0_seed17_mem8g",
        task_id="g2-dev-core50-nc-s0-seed17-mem8g",
        notes="G2 mem8g CORe50-NC S0 seed=17. Not a freeze.",
    )
    write_task(
        "g2-occupancy-mem8g",
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    )
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "note": "8G admission calibration. S0 twice + high-resource static. kind=probe only.",
    }
    (TASK / "_batch_g2_dev_mem8g.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(TASK_IDS))


if __name__ == "__main__":
    main()
