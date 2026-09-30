"""Clone remaining G2 streams onto mem16g. Does not overwrite earlier mem16g CIFAR100/NC runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem16g-b"
SEQ = 600
CAP = "mem16g"


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
        "required_capacity": CAP,
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
    text = text.replace("board_capacity_id: mem64g", f"board_capacity_id: {CAP}")
    text = text.replace("board_capacity_id: mem32g", f"board_capacity_id: {CAP}")
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
    jobs = [
        ("g2_cifar10_s0_seed17.yaml", "g2_cifar10_s0_seed17_mem16g", "fullmem_v4_g2_cifar10_s0_seed17_mem16g", "S0", "G2 mem16g CIFAR10 S0 seed=17. Not a freeze.", "g2-dev-cifar10-s0-seed17-mem16g"),
        ("g2_cifar10_gss_seed17.yaml", "g2_cifar10_gss_seed17_mem16g", "fullmem_v4_g2_cifar10_gss_seed17_mem16g", None, "G2 mem16g CIFAR10 GSS seed=17. Not a freeze.", "g2-dev-cifar10-gss-seed17-mem16g"),
        ("g2_cifar10_gem_seed17.yaml", "g2_cifar10_gem_seed17_mem16g", "fullmem_v4_g2_cifar10_gem_seed17_mem16g", None, "G2 mem16g CIFAR10 GEM seed=17. Not a freeze.", "g2-dev-cifar10-gem-seed17-mem16g"),
        ("g2_cifar10_agem_seed17.yaml", "g2_cifar10_agem_seed17_mem16g", "fullmem_v4_g2_cifar10_agem_seed17_mem16g", None, "G2 mem16g CIFAR10 AGEM seed=17. Not a freeze.", "g2-dev-cifar10-agem-seed17-mem16g"),
        ("g2_cifar10_er_gem_seed17.yaml", "g2_cifar10_er_gem_seed17_mem16g", "fullmem_v4_g2_cifar10_er_gem_seed17_mem16g", None, "G2 mem16g CIFAR10 ER+GEM seed=17. Not a freeze.", "g2-dev-cifar10-er-gem-seed17-mem16g"),
        ("g2_cifar10_er_ewc_seed17.yaml", "g2_cifar10_er_ewc_seed17_mem16g", "fullmem_v4_g2_cifar10_er_ewc_seed17_mem16g", None, "G2 mem16g CIFAR10 ER+EWC seed=17. Not a freeze.", "g2-dev-cifar10-er-ewc-seed17-mem16g"),
        ("g2_cifar10_er_gem_ewc_seed17.yaml", "g2_cifar10_er_gem_ewc_seed17_mem16g", "fullmem_v4_g2_cifar10_er_gem_ewc_seed17_mem16g", None, "G2 mem16g CIFAR10 ER+GEM+EWC seed=17. Not a freeze.", "g2-dev-cifar10-er-gem-ewc-seed17-mem16g"),
        ("g2_cifar10_lr_seed17.yaml", "g2_cifar10_lr_seed17_mem16g", "fullmem_v4_g2_cifar10_lr_seed17_mem16g", None, "G2 mem16g CIFAR10 LR reconstructed seed=17. Not a freeze.", "g2-dev-cifar10-lr-seed17-mem16g"),
        ("g2_core50_ni_s0_seed17.yaml", "g2_core50_ni_s0_seed17_mem16g", "fullmem_v4_g2_core50_ni_s0_seed17_mem16g", None, "G2 mem16g CORe50-NI S0 seed=17. Not a freeze.", "g2-dev-core50-ni-s0-seed17-mem16g"),
        ("g2_core50_ni_er_gem_seed17.yaml", "g2_core50_ni_er_gem_seed17_mem16g", "fullmem_v4_g2_core50_ni_er_gem_seed17_mem16g", None, "G2 mem16g CORe50-NI ER+GEM seed=17. Not a freeze.", "g2-dev-core50-ni-er-gem-seed17-mem16g"),
        ("g2_core50_ni_er_ewc_seed17.yaml", "g2_core50_ni_er_ewc_seed17_mem16g", "fullmem_v4_g2_core50_ni_er_ewc_seed17_mem16g", None, "G2 mem16g CORe50-NI ER+EWC seed=17. Not a freeze.", "g2-dev-core50-ni-er-ewc-seed17-mem16g"),
        ("g2_core50_ni_er_gem_ewc_seed17.yaml", "g2_core50_ni_er_gem_ewc_seed17_mem16g", "fullmem_v4_g2_core50_ni_er_gem_ewc_seed17_mem16g", None, "G2 mem16g CORe50-NI ER+GEM+EWC seed=17. Not a freeze.", "g2-dev-core50-ni-er-gem-ewc-seed17-mem16g"),
        ("g2_core50_ni_lr_seed17.yaml", "g2_core50_ni_lr_seed17_mem16g", "fullmem_v4_g2_core50_ni_lr_seed17_mem16g", None, "G2 mem16g CORe50-NI LR reconstructed seed=17. Not a freeze.", "g2-dev-core50-ni-lr-seed17-mem16g"),
        ("g2_core50_nic_s0_seed17.yaml", "g2_core50_nic_s0_seed17_mem16g", "fullmem_v4_g2_core50_nic_s0_seed17_mem16g", None, "G2 mem16g CORe50-NIC S0 nicv2_79 seed=17. Not plugin stack freeze.", "g2-dev-core50-nic-s0-seed17-mem16g"),
        ("g2_core50_nic_lr_seed17.yaml", "g2_core50_nic_lr_seed17_mem16g", "fullmem_v4_g2_core50_nic_lr_seed17_mem16g", None, "G2 mem16g CORe50-NIC LR reconstructed seed=17. Not a freeze.", "g2-dev-core50-nic-lr-seed17-mem16g"),
        ("g2_core50_nic_er_gem_ewc_seed17.yaml", "g2_core50_nic_er_gem_ewc_seed17_mem16g", "fullmem_v4_g2_core50_nic_er_gem_ewc_seed17_mem16g", None, "G2 mem16g CORe50-NIC ER+GEM+EWC nicv2_79 seed=17. Plugin-stack probe, not a freeze.", "g2-dev-core50-nic-er-gem-ewc-seed17-mem16g"),
    ]
    for src, stem, run_id, method_id, notes, task_id in jobs:
        clone(
            src,
            stem=stem,
            run_id=run_id,
            experiment_id="V4_g2_dev_mem16g_b",
            method_id=method_id,
            notes=notes,
        )
        write_task(
            task_id,
            next_seq(),
            ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
        )
    write_task(
        "g2-occupancy-mem16g-b",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    )
    write_task(
        "g2-idle-mem16g-b",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"],
    )


if __name__ == "__main__":
    main()
