"""Emit the next H4 development chunk.

- NC BR-adapt low→high→low retry (new task id; original implementation_error kept).
- NIC BR-adapt both sequences (plugins stay off).
- Dynamic-S* static search on the actual H4 sequences: five candidates other than
  b16/r200, which is the already completed S0 cell. NC and NIC, both sequences.
- Occupancy after the search.

kind=probe. Not a freeze. Not kind=train. b16/r200 is not re-emitted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-h4-sstar"
SEQ = 1400
CAP = "mem8g"
EXPERIMENT = "V4_g2_dev_h4_sstar_mem8g"
CANDIDATES = ((16, 2000), (64, 200), (64, 2000), (256, 200), (256, 2000))
SEQUENCES = ("low_high_low", "high_low_high")
NIC_L_CAL = 1.871

NIC_DATASET = (
    "dataset:\n"
    "  name: core50_nic\n"
    "  root: data/raw/core50\n"
    "  split_dir: reports/fullmem_v4/data/core50_nicv2_79_run0_dev_split_seed17\n"
    "  split_manifest: data/manifests/core50_nicv2_79_run0_dev_split_seed17.json\n"
    "  n_experiences: 79\n"
    "  experience_limit: 79\n"
    "  shuffle: false\n"
    "  scenario: nicv2_79\n"
    "  run: 0\n"
    "  object_level: true\n"
)

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


def write_yaml(text: str, stem: str) -> None:
    dest = CFG / f"{stem}.yaml"
    if dest.exists():
        raise SystemExit(f"refuse overwrite {dest.name}")
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)


def retarget(text: str, *, run_id: str, method_id: str, notes: str) -> str:
    text = re.sub(r"^run_id: .*$", f"run_id: {run_id}", text, count=1, flags=re.M)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {EXPERIMENT}", text, count=1, flags=re.M)
    text = re.sub(r"^method_id: .*$", f"method_id: {method_id}", text, count=1, flags=re.M)
    text = re.sub(r"^notes:.*$", f"notes: {json.dumps(notes)}", text, count=1, flags=re.M)
    return text


def set_sequence(text: str, sequence: str) -> str:
    return re.sub(r"^  sequence: .*$", f"  sequence: {sequence}", text, count=1, flags=re.M)


def br_adapt(text: str, *, sequence: str, latency_s: float | None, notes: str, stem: str) -> None:
    run_id = f"fullmem_v4_{stem}"
    text = retarget(text, run_id=run_id, method_id="BR-adapt", notes=notes)
    text = set_sequence(text, sequence)
    text = re.sub(r"^  optional_plugins: .*$", "  optional_plugins: none", text, count=1, flags=re.M)
    text = re.sub(r"^  plugin_policy: .*$", "  plugin_policy: fixed_default", text, count=1, flags=re.M)
    if latency_s is not None:
        text = re.sub(r"^    latency_s: .*$", f"    latency_s: {latency_s}", text, count=1, flags=re.M)
    if "plugin_policy: fixed_default" not in text:
        raise SystemExit(f"{stem} missing fixed_default")
    write_yaml(text, stem)
    task_id = "g2-dev-" + stem.replace("g2_", "").replace("_", "-")
    write_task(task_id, ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"])


def sstar(src_name: str, *, dataset: str, batch: int, replay: int, sequence: str) -> None:
    stem = f"g2_{dataset}_sstar_b{batch}_r{replay}_h4_{sequence}_seed17_mem8g"
    text = (CFG / src_name).read_text(encoding="utf-8")
    notes = (
        f"G2 H4 {dataset} static S* candidate b{batch}/r{replay} {sequence}. "
        "Controller off, plugins none, replay_batch equals new_batch. "
        "Search is on this H4 sequence, not the 64G CIFAR100 choice. Not frozen. Not formal."
    )
    text = retarget(text, run_id=f"fullmem_v4_{stem}", method_id="Sstar_candidate", notes=notes)
    text = set_sequence(text, sequence)
    text = re.sub(r"^  new_batch: .*$", f"  new_batch: {batch}", text, count=1, flags=re.M)
    text = re.sub(r"^  replay_batch: .*$", f"  replay_batch: {batch}", text, count=1, flags=re.M)
    text = re.sub(r"^  capacity: .*$", f"  capacity: {replay}", text, count=1, flags=re.M)
    if "controller:\n  enabled: false" not in text and "controller:\r\n  enabled: false" not in text:
        if not re.search(r"^controller:\n  enabled: false$", text, flags=re.M):
            raise SystemExit(f"{stem} controller must stay off")
    write_yaml(text, stem)
    task_id = "g2-dev-" + stem.replace("g2_", "").replace("_", "-")
    write_task(task_id, ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"])


def main() -> None:
    TASK.mkdir(parents=True, exist_ok=True)
    br_src = (CFG / "g2_core50_nc_br_adapt_h4_low_high_low_seed17_mem8g.yaml").read_text(encoding="utf-8")
    br_adapt(
        br_src,
        sequence="low_high_low",
        latency_s=None,
        notes=(
            "G2 H4 NC BR-adapt low_high_low retry b2. Original attempt kept as implementation_error. "
            "plugin_policy fixed_default: batch/replay adapt, plugins stay off. Not O-recon. Not formal."
        ),
        stem="g2_core50_nc_br_adapt_h4_low_high_low_seed17_mem8g_b2",
    )
    nic_br = re.sub(r"^dataset:\n(?:  .*\n)+", NIC_DATASET, br_src, count=1, flags=re.M)
    nic_br = nic_br.replace("v3_dataset: core50_nc", "v3_dataset: core50_nic")
    for sequence in SEQUENCES:
        br_adapt(
            nic_br,
            sequence=sequence,
            latency_s=NIC_L_CAL,
            notes=(
                f"G2 H4 NIC BR-adapt {sequence}. L_cal={NIC_L_CAL}s. "
                "plugin_policy fixed_default: batch/replay adapt, plugins stay off. Not O-recon. Not formal."
            ),
            stem=f"g2_core50_nic_br_adapt_h4_{sequence}_seed17_mem8g",
        )
    for dataset, prefix in (("core50_nc", "g2_core50_nc_s0"), ("core50_nic", "g2_core50_nic_s0")):
        for batch, replay in CANDIDATES:
            for sequence in SEQUENCES:
                src = f"{prefix}_h4_{sequence}_seed17_mem8g.yaml"
                sstar(src, dataset=dataset, batch=batch, replay=replay, sequence=sequence)
    write_task("g2-occupancy-h4-sstar", ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"])
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "reused_s0_as_b16_r200": True,
        "not_emitted": ["b16_r200_already_completed_as_h4_s0", "nic_orecon_retries"],
        "note": "S* search is static on the H4 sequence. BR-adapt keeps plugins off.",
    }
    (TASK / "_batch_g2_dev_h4_sstar.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("batch", BATCH, "n", len(TASK_IDS))
    for task in TASK_IDS:
        print(task)


if __name__ == "__main__":
    main()
