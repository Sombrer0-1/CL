"""Formal static comparison at the mem5g tight candidate. kind=train.

CIFAR-100 and CORe50-NC, S0 / static S* b16/r2000 / O-recon, seeds 0/1/2.
No H4 holder. S* is the recorded development selection, not a new search on 5G.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
FREEZE = ROOT / "reports" / "fullmem_v4" / "g3" / "FREEZE.json"
BATCH = "g4-static-tight-mem5g"
EXPERIMENT = "V4_g4_static_tight_mem5g"
SEQ = 3000
SEEDS = (0, 1, 2)


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def retarget(text: str, *, run_id: str, method_id: str, seed: int, notes: str) -> str:
    text = re.sub(r"^run_id: .*$", f"run_id: {run_id}", text, count=1, flags=re.M)
    text = re.sub(r"^phase: .*$", "phase: formal", text, count=1, flags=re.M)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^protocol_id: .*$", "protocol_id: fullmem_v4_g4_static_tight", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^method_id: .*$", f"method_id: {method_id}", text, count=1, flags=re.M)
    text = re.sub(r"^  model: .*$", f"  model: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  stream: .*$", f"  stream: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  replay: .*$", f"  replay: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  augmentation: .*$", f"  augmentation: {seed}", text, count=1, flags=re.M)
    text = text.replace("board_capacity_id: mem64g", "board_capacity_id: mem5g")
    text = text.replace("board_capacity_id: mem8g", "board_capacity_id: mem5g")
    text = text.replace("board_capacity_id: mem16g", "board_capacity_id: mem5g")
    text = re.sub(r"^notes:.*$", "notes: " + json.dumps(notes), text, count=1, flags=re.M)
    if "resource_envelope:" in text:
        raise SystemExit(f"{run_id} still has a resource envelope")
    return text


def write_task(task_id: str, yaml_name: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": next_seq(),
        "kind": "train",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.run",
            "--config",
            f"configs/fullmem_v4/{yaml_name}",
        ],
        "required_capacity": "mem5g",
        "batch_id": BATCH,
    }
    dest = TASK / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    backup = FREEZE.with_name("FREEZE_h4_mem8g.json")
    if not backup.exists():
        shutil.copyfile(FREEZE, backup)
    admitted = list(freeze.get("admitted_batches") or [])
    if BATCH not in admitted:
        admitted.append(BATCH)
    freeze["admitted_batches"] = admitted
    freeze["capacity_id"] = "mem5g"
    freeze["static_tight_candidate"] = "mem5g"
    freeze["static_loose_candidate"] = "mem8g"
    freeze["static_mid"] = "unestablished"
    freeze["mem5g_note"] = (
        "S0 completed twice. CIFAR100 batch256 min MemAvailable was 194809856 bytes. "
        "No resource failure. factor_m still saturated above 32 MiB. Not a three-tier freeze."
    )
    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8", newline="\n")

    specs = [
        ("cifar100", "s0", "S0", "g2_cifar100_s0_seed17_mem5g.yaml", None),
        ("cifar100", "sstar", "Sstar", "g2_cifar100_s0_seed17_mem5g.yaml", 2000),
        ("cifar100", "orecon", "O-recon", "g2_cifar100_orecon_full.yaml", None),
        ("core50_nc", "s0", "S0", "g2_core50_nc_s0_seed17_mem5g.yaml", None),
        ("core50_nc", "sstar", "Sstar", "g2_core50_nc_s0_seed17_mem5g.yaml", 2000),
        ("core50_nc", "orecon", "O-recon", "g2_core50_nc_orecon_full.yaml", None),
    ]
    tasks: list[str] = []
    for seed in SEEDS:
        for dataset, method, method_id, src, replay in specs:
            stem = f"g4_{dataset}_{method}_static_seed{seed}_mem5g"
            text = (CFG / src).read_text(encoding="utf-8")
            if replay is not None:
                old = "capacity: 200\n"
                new = "capacity: 2000\n"
                if old not in text:
                    raise SystemExit(f"{src} has no replay capacity 200")
                text = text.replace(old, new, 1)
                if "enabled: false" not in text:
                    raise SystemExit(f"{src} controller/prefetch not disabled for S*")
            notes = (
                f"Formal static tight candidate mem5g {dataset} {method_id} seed {seed}. "
                "No H4 holder. S* is development b16/r2000, not re-searched on 5G. "
                "8G remains the loose candidate. Mid unestablished."
            )
            text = retarget(text, run_id=f"fullmem_v4_{stem}", method_id=method_id, seed=seed, notes=notes)
            yaml_name = f"{stem}.yaml"
            (CFG / yaml_name).write_text(text, encoding="utf-8", newline="\n")
            task_id = stem.replace("_", "-")
            # stem uses underscores; task ids in this project use hyphens.
            task_id = f"g4-{dataset.replace('_', '-')}-{method}-static-seed{seed}-mem5g"
            write_task(task_id, yaml_name)
            tasks.append(task_id)
            print("yaml", yaml_name)
    man = {
        "batch_id": BATCH,
        "kind": "train",
        "required_capacity": "mem5g",
        "n_tasks": len(tasks),
        "tasks": tasks,
        "note": "Static tight-candidate main comparison. Not the 1320-cell matrix.",
    }
    (TASK / "_batch_g4_static_tight_mem5g.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(tasks))


if __name__ == "__main__":
    main()
