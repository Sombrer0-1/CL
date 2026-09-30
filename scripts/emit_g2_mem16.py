"""Clone G2 profiling yamls onto mem16g. Does not overwrite mem64g/mem32g runs. Not a freeze."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem16g"
SEQ = 500
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
        (
            "g1_cifar100_exp1.yaml",
            "g1_cifar100_exp1_mem16g",
            "fullmem_v4_g1_cifar100_exp1_mem16g",
            "V4_g1_board_mem16g_real_data",
            None,
            "G1 mem16g one-experience S0. Not a freeze.",
            "g1-cifar100-exp1-mem16g",
        ),
        (
            "g2_cifar100_s0_hash_off_seed17.yaml",
            "g2_cifar100_s0_seed17_mem16g",
            "fullmem_v4_g2_cifar100_s0_seed17_mem16g",
            "V4_g2_dev_mem16g",
            "S0",
            "G2 mem16g CIFAR100 S0 seed=17. Not a freeze.",
            "g2-dev-cifar100-s0-seed17-mem16g",
        ),
        (
            "g2_cifar100_gss_seed17.yaml",
            "g2_cifar100_gss_seed17_mem16g",
            "fullmem_v4_g2_cifar100_gss_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 GSS seed=17. Not a freeze.",
            "g2-dev-cifar100-gss-seed17-mem16g",
        ),
        (
            "g2_cifar100_gem_seed17.yaml",
            "g2_cifar100_gem_seed17_mem16g",
            "fullmem_v4_g2_cifar100_gem_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 GEM seed=17. Not a freeze.",
            "g2-dev-cifar100-gem-seed17-mem16g",
        ),
        (
            "g2_cifar100_agem_seed17.yaml",
            "g2_cifar100_agem_seed17_mem16g",
            "fullmem_v4_g2_cifar100_agem_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 AGEM seed=17. Not a freeze.",
            "g2-dev-cifar100-agem-seed17-mem16g",
        ),
        (
            "g2_cifar100_er_gem_seed17.yaml",
            "g2_cifar100_er_gem_seed17_mem16g",
            "fullmem_v4_g2_cifar100_er_gem_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 ER+GEM seed=17. Not a freeze.",
            "g2-dev-cifar100-er-gem-seed17-mem16g",
        ),
        (
            "g2_cifar100_er_ewc_seed17.yaml",
            "g2_cifar100_er_ewc_seed17_mem16g",
            "fullmem_v4_g2_cifar100_er_ewc_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 ER+EWC seed=17. Not a freeze.",
            "g2-dev-cifar100-er-ewc-seed17-mem16g",
        ),
        (
            "g2_cifar100_er_gem_ewc_seed17.yaml",
            "g2_cifar100_er_gem_ewc_seed17_mem16g",
            "fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 ER+GEM+EWC seed=17. Not a freeze.",
            "g2-dev-cifar100-er-gem-ewc-seed17-mem16g",
        ),
        (
            "g2_cifar100_lr_seed17.yaml",
            "g2_cifar100_lr_seed17_mem16g",
            "fullmem_v4_g2_cifar100_lr_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 LR reconstructed seed=17. Not a freeze.",
            "g2-dev-cifar100-lr-seed17-mem16g",
        ),
        (
            "g2_cifar100_sstar_b16_r2000_seed17.yaml",
            "g2_cifar100_sstar_b16_r2000_seed17_mem16g",
            "fullmem_v4_g2_cifar100_sstar_b16_r2000_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 S* b16/r2000 seed=17. Development choice, not a freeze.",
            "g2-dev-cifar100-sstar-b16-r2000-seed17-mem16g",
        ),
        (
            "g2_cifar100_orecon_full_hash_off_seed17.yaml",
            "g2_cifar100_orecon_full_hash_off_seed17_mem16g",
            "fullmem_v4_g2_cifar100_orecon_full_hash_off_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CIFAR100 O-recon full hash-off seed=17. Not a freeze.",
            "g2-dev-cifar100-orecon-full-hash-off-seed17-mem16g",
        ),
        (
            "g2_cifar100_batch256.yaml",
            "g2_cifar100_batch256_seed17_mem16g",
            "fullmem_v4_g2_cifar100_batch256_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g high-batch pressure identifiability, seed=17.",
            "g2-dev-cifar100-batch256-seed17-mem16g",
        ),
        (
            "g2_cifar100_replay2000.yaml",
            "g2_cifar100_replay2000_seed17_mem16g",
            "fullmem_v4_g2_cifar100_replay2000_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g high-replay pressure identifiability, seed=17.",
            "g2-dev-cifar100-replay2000-seed17-mem16g",
        ),
        (
            "g2_core50_nc_s0_seed17.yaml",
            "g2_core50_nc_s0_seed17_mem16g",
            "fullmem_v4_g2_core50_nc_s0_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CORe50-NC S0 seed=17. Not a freeze.",
            "g2-dev-core50-nc-s0-seed17-mem16g",
        ),
        (
            "g2_core50_nc_gss_seed17.yaml",
            "g2_core50_nc_gss_seed17_mem16g",
            "fullmem_v4_g2_core50_nc_gss_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CORe50-NC GSS seed=17. Not a freeze.",
            "g2-dev-core50-nc-gss-seed17-mem16g",
        ),
        (
            "g2_core50_nc_er_gem_ewc_seed17.yaml",
            "g2_core50_nc_er_gem_ewc_seed17_mem16g",
            "fullmem_v4_g2_core50_nc_er_gem_ewc_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CORe50-NC ER+GEM+EWC seed=17. Not a freeze.",
            "g2-dev-core50-nc-er-gem-ewc-seed17-mem16g",
        ),
        (
            "g2_core50_nc_orecon_full_hash_off_seed17.yaml",
            "g2_core50_nc_orecon_full_hash_off_seed17_mem16g",
            "fullmem_v4_g2_core50_nc_orecon_full_hash_off_seed17_mem16g",
            "V4_g2_dev_mem16g",
            None,
            "G2 mem16g CORe50-NC O-recon full hash-off seed=17. Not a freeze.",
            "g2-dev-core50-nc-orecon-full-hash-off-seed17-mem16g",
        ),
    ]
    for src, stem, run_id, experiment_id, method_id, notes, task_id in jobs:
        clone(
            src,
            stem=stem,
            run_id=run_id,
            experiment_id=experiment_id,
            method_id=method_id,
            notes=notes,
        )
        write_task(
            task_id,
            next_seq(),
            ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
        )
    write_task(
        "g2-occupancy-mem16g",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
    )
    write_task(
        "g2-idle-mem16g",
        next_seq(),
        ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"],
    )


if __name__ == "__main__":
    main()
