"""Emit the frozen formal H4 block: 48 kind=train tasks.

NC/NIC × S0 / sequence-specific S* / O-recon / BR-adapt × 2 sequences × seeds 0/1/2.
Data splits stay the registered seed-17 manifests. Requires reports/fullmem_v4/g3/FREEZE.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
FREEZE = ROOT / "reports" / "fullmem_v4" / "g3" / "FREEZE.json"
BATCH = "g4-h4-mem8g"
EXPERIMENT = "V4_g4_h4_mem8g"
SEQ = 2000
SEEDS = (0, 1, 2)
SEQUENCES = ("low_high_low", "high_low_high")
METHODS = ("s0", "sstar", "br-adapt", "orecon")

TEMPLATES = {
    ("core50_nc", "s0"): "g2_core50_nc_s0_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nc", "sstar"): "g2_core50_nc_s0_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nc", "br-adapt"): "g2_core50_nc_br_adapt_h4_low_high_low_seed17_mem8g_b2.yaml",
    ("core50_nc", "orecon"): "g2_core50_nc_orecon_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nic", "s0"): "g2_core50_nic_s0_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nic", "sstar"): "g2_core50_nic_s0_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nic", "br-adapt"): "g2_core50_nic_br_adapt_h4_low_high_low_seed17_mem8g.yaml",
    ("core50_nic", "orecon"): "g2_core50_nic_orecon_h4_low_high_low_seed17_mem8g.yaml",
}
METHOD_IDS = {"s0": "S0", "sstar": "Sstar", "br-adapt": "BR-adapt", "orecon": "O-recon"}


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def retarget(text: str, *, run_id: str, method_id: str, seed: int, sequence: str, notes: str, freeze: dict, manifest_key: str) -> str:
    text = re.sub(r"^run_id: .*$", f"run_id: {run_id}", text, count=1, flags=re.M)
    text = re.sub(r"^phase: .*$", "phase: formal", text, count=1, flags=re.M)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^protocol_id: .*$", "protocol_id: fullmem_v4_g4_h4", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^method_id: .*$", f"method_id: {method_id}", text, count=1, flags=re.M)
    text = re.sub(r"^dataset_manifest_sha256: .*$", f"dataset_manifest_sha256: {freeze['dataset_manifest_sha256'][manifest_key]}", text, count=1, flags=re.M)
    text = re.sub(r"^code_revision_or_snapshot: .*$", f"code_revision_or_snapshot: {freeze['source_snapshot_sha256']}", text, count=1, flags=re.M)
    text = re.sub(r"^environment_lock_sha256: .*$", f"environment_lock_sha256: {freeze['environment_lock_sha256']}", text, count=1, flags=re.M)
    text = re.sub(r"^  model: .*$", f"  model: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  stream: .*$", f"  stream: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  replay: .*$", f"  replay: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  augmentation: .*$", f"  augmentation: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  sequence: .*$", f"  sequence: {sequence}", text, count=1, flags=re.M)
    text = re.sub(r"^notes:.*$", "notes: " + json.dumps(notes), text, count=1, flags=re.M)
    return text


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("admitted_batches") != [BATCH]:
        raise SystemExit("freeze does not admit this batch")
    if int(freeze["h4_hold_bytes"]) != 2684354560:
        raise SystemExit("freeze hold changed")
    TASK.mkdir(parents=True, exist_ok=True)
    tasks: list[str] = []
    yamls: list[str] = []
    for seed in SEEDS:
        for dataset in ("core50_nc", "core50_nic"):
            manifest_key = (
                "data/manifests/core50_nc_run0_dev_split_seed17.json"
                if dataset == "core50_nc"
                else "data/manifests/core50_nicv2_79_run0_dev_split_seed17.json"
            )
            for sequence in SEQUENCES:
                for method in METHODS:
                    chosen = freeze["dynamic_sstar"][dataset][sequence]
                    stem = f"g4_{dataset}_{method}_h4_{sequence}_seed{seed}_mem8g"
                    run_id = f"fullmem_v4_{stem}"
                    text = (CFG / TEMPLATES[(dataset, method)]).read_text(encoding="utf-8")
                    notes = (
                        f"Formal H4 {dataset} {method} {sequence} seed {seed}. "
                        f"Split remains seed 17. Hold 2.5GiB. Not tight. "
                        f"S* for this sequence is b{chosen['new_batch']}/r{chosen['replay_capacity']}."
                    )
                    text = retarget(
                        text,
                        run_id=run_id,
                        method_id=METHOD_IDS[method],
                        seed=seed,
                        sequence=sequence,
                        notes=notes,
                        freeze=freeze,
                        manifest_key=manifest_key,
                    )
                    if method == "sstar":
                        batch = int(chosen["new_batch"])
                        replay = int(chosen["replay_capacity"])
                        text = re.sub(r"^  new_batch: .*$", f"  new_batch: {batch}", text, count=1, flags=re.M)
                        text = re.sub(r"^  replay_batch: .*$", f"  replay_batch: {batch}", text, count=1, flags=re.M)
                        text = re.sub(r"^  capacity: .*$", f"  capacity: {replay}", text, count=1, flags=re.M)
                        if "enabled: false" not in text.split("controller:", 1)[1][:80]:
                            raise SystemExit(f"{stem} S* controller must stay off")
                    dest = CFG / f"{stem}.yaml"
                    if dest.exists():
                        raise SystemExit(f"refuse overwrite {dest.name}")
                    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
                    yamls.append(dest.name)
                    task_id = "g4-h4-" + stem.replace("g4_", "").replace("_", "-")
                    payload = {
                        "study_id": "fullmem_v4",
                        "design_version": "design-v1",
                        "task_id": task_id,
                        "seq": next_seq(),
                        "kind": "train",
                        "argv": ["/home/zhuzetong/miniconda3/envs/orion/bin/python", "-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{dest.name}"],
                        "required_capacity": "mem8g",
                        "batch_id": BATCH,
                    }
                    (TASK / f"{task_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
                    tasks.append(task_id)
    if len(tasks) != 48:
        raise SystemExit(f"expected 48 tasks, got {len(tasks)}")
    man = {"batch_id": BATCH, "kind": "train", "n_tasks": 48, "tasks": tasks, "yamls": yamls, "order_frozen": True}
    (TASK / "_batch_g4_h4_mem8g.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(tasks))


if __name__ == "__main__":
    main()
