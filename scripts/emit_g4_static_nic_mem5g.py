"""Formal static NIC comparison at mem5g. kind=train. No H4 holder.

S0 and S* come from the NIC S0 yaml. S* sets replay capacity 2000 and leaves
the controller off. O-recon starts from the reviewed H4 NIC O-recon yaml so
the NIC L_cal stays, then drops the background holder and turns prefetch on
to match the static full method used for NC.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g4-static-tight-mem5g"
EXPERIMENT = "V4_g4_static_tight_mem5g"
SEQ = 3200
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
    text = text.replace("feedback_source: development_val_seen", "feedback_source: official_test_seen")
    text = text.replace("board_capacity_id: mem16g", "board_capacity_id: mem5g")
    text = text.replace("board_capacity_id: mem8g", "board_capacity_id: mem5g")
    text = re.sub(r"^notes:.*$", "notes: " + json.dumps(notes), text, count=1, flags=re.M)
    if "feedback_source: official_test_seen" not in text:
        raise SystemExit(f"{run_id} missing official_test_seen")
    if "resource_envelope:" in text:
        raise SystemExit(f"{run_id} still has a resource envelope")
    return text


def strip_envelope(text: str) -> str:
    lines = text.splitlines()
    out = []
    skipping = False
    for line in lines:
        if line.startswith("resource_envelope:"):
            skipping = True
            continue
        if skipping:
            if line.startswith(" ") or line.startswith("\t"):
                continue
            skipping = False
        out.append(line)
    return "\n".join(out) + "\n"


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
    (TASK / f"{task_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    freeze = json.loads((ROOT / "reports/fullmem_v4/g3/FREEZE.json").read_text(encoding="utf-8"))
    if BATCH not in (freeze.get("admitted_batches") or []):
        raise SystemExit("batch not admitted")
    if freeze.get("capacity_id") != "mem5g":
        raise SystemExit("freeze capacity is not mem5g")
    s0 = (CFG / "g2_core50_nic_s0_seed17_mem16g.yaml").read_text(encoding="utf-8")
    ore = strip_envelope((CFG / "g2_core50_nic_orecon_h4_low_high_low_seed17_mem8g.yaml").read_text(encoding="utf-8"))
    ore = ore.replace("enabled: false\n  queue_depth: 1", "enabled: true\n  queue_depth: 2", 1)
    if "prefetch:\n  enabled: true" not in ore and "enabled: true" not in ore.split("prefetch:", 1)[-1][:80]:
        raise SystemExit("failed to enable prefetch on NIC O-recon")
    tasks = []
    for seed in SEEDS:
        specs = [
            ("s0", "S0", s0, False),
            ("sstar", "Sstar", s0, True),
            ("orecon", "O-recon", ore, False),
        ]
        for method, method_id, src, is_sstar in specs:
            text = src
            if is_sstar:
                if "capacity: 200\n" not in text:
                    raise SystemExit("NIC S0 missing replay capacity 200")
                text = text.replace("capacity: 200\n", "capacity: 2000\n", 1)
            stem = f"g4_core50_nic_{method}_static_seed{seed}_mem5g"
            notes = (
                f"Formal static tight candidate mem5g core50_nic {method_id} seed {seed}. "
                "No H4 holder. S* is development b16/r2000. "
                "O-recon keeps the NIC L_cal from the H4 yaml and enables prefetch like the NC static full method."
            )
            text = retarget(text, run_id=f"fullmem_v4_{stem}", method_id=method_id, seed=seed, notes=notes)
            yaml_name = f"{stem}.yaml"
            (CFG / yaml_name).write_text(text, encoding="utf-8", newline="\n")
            task_id = f"g4-core50-nic-{method}-static-seed{seed}-mem5g"
            write_task(task_id, yaml_name)
            tasks.append(task_id)
            print("yaml", yaml_name)
    (TASK / "_batch_g4_static_nic_mem5g.json").write_text(
        json.dumps({"batch_id": BATCH, "tasks": tasks}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("n", len(tasks))


if __name__ == "__main__":
    main()
