"""Emit remaining H4 development methods: NIC O-recon and NC BR-adapt.

kind=probe. Not a freeze. Not 48-slot formal. Dynamic S* is not emitted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
CAL = ROOT / "reports" / "fullmem_v4" / "g2" / "h4_calibrate.json"
LCAL_NIC = ROOT / "reports" / "fullmem_v4" / "g2" / "l_cal_nic.json"
BATCH = "g2-dev-h4-more"
SEQ = 1300
CAP = "mem8g"
SRC_ORECON = "g2_core50_nc_orecon_full_hash_off_seed17_mem16g.yaml"
NIC_L_CAL = 1.871
CHUNK = 256 * 1024 * 1024
TASK_IDS: list[str] = []
YAML_NAMES: list[str] = []

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


def training_high_bytes(cal: dict) -> int:
    calibrated = int(cal.get("high_bytes") or 0)
    if calibrated <= 0:
        raise SystemExit(f"refuse emit: calibration high_bytes={calibrated}")
    baseline = int(cal.get("baseline_mem_available_bytes") or 0)
    os_floor = 1024 * 1024 * 1024
    train_headroom = 1536 * 1024 * 1024
    training_cap = max(0, baseline - os_floor - train_headroom) if baseline else calibrated
    high_bytes = min(calibrated, training_cap)
    high_bytes = (high_bytes // CHUNK) * CHUNK
    if high_bytes < 1024 * 1024 * 1024:
        raise SystemExit(f"refuse emit: training hold {high_bytes} < 1GiB identifiability")
    return high_bytes


def inject_envelope(text: str, *, sequence: str, high_bytes: int, notes: str) -> str:
    block = (
        "resource_envelope:\n"
        "  enabled: true\n"
        "  mode: background_process\n"
        f"  sequence: {sequence}\n"
        f"  high_bytes: {int(high_bytes)}\n"
        "  low_bytes: 0\n"
    )
    if "resource_envelope:" in text:
        raise SystemExit("source already has resource_envelope")
    text = text.replace("\nnotes:", "\n" + block + "notes:", 1)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("notes:"):
            lines[i] = f"notes: {json.dumps(notes)}"
    return "\n".join(lines) + "\n"


def retarget_h4(text: str, *, experiment_id: str) -> str:
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit("no run_id")
    text = text.replace(f"run_id: {old_run}", "run_id: PLACEHOLDER_RUN", 1)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {experiment_id}", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {experiment_id}", text, count=1, flags=re.M)
    for old in ("mem64g", "mem32g", "mem16g"):
        text = text.replace(f"board_capacity_id: {old}", f"board_capacity_id: {CAP}")
        text = text.replace(f"pressure_role: g2_dev_calibration_{old}", "pressure_role: g2_dev_h4_mem8g")
    text = text.replace("pressure_role: g2_dev_calibration_mem8g", "pressure_role: g2_dev_h4_mem8g")
    text = re.sub(r"^  controlled_resource: .*$", "  controlled_resource: board", text, count=1, flags=re.M)
    return text


def to_nic_orecon(text: str, latency_s: float) -> str:
    text = re.sub(r"^dataset:\n(?:  .*\n)+", NIC_DATASET, text, count=1, flags=re.M)
    text = text.replace("v3_dataset: core50_nc", "v3_dataset: core50_nic")
    text = re.sub(r"^    latency_s: .*$", f"    latency_s: {latency_s}", text, count=1, flags=re.M)
    return text


def to_br_adapt(text: str) -> str:
    text = re.sub(r"^method_id: .*$", "method_id: BR-adapt", text, count=1, flags=re.M)
    text = re.sub(r"^  optional_plugins: .*$", "  optional_plugins: none", text, count=1, flags=re.M)
    text = re.sub(r"^  plugin_policy: .*$", "  plugin_policy: fixed_default", text, count=1, flags=re.M)
    return text


def write_yaml(text: str, *, stem: str, sequence: str, high_bytes: int, notes: str) -> None:
    run_id = f"fullmem_v4_{stem}"
    text = text.replace("run_id: PLACEHOLDER_RUN", f"run_id: {run_id}", 1)
    text = inject_envelope(text, sequence=sequence, high_bytes=high_bytes, notes=notes)
    dest = CFG / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)
    print("yaml", dest.name)
    task_id = "g2-dev-" + stem.replace("g2_", "").replace("_", "-")
    write_task(task_id, ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{dest.name}"])


def main() -> None:
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    high_bytes = training_high_bytes(cal)
    latency_nic = NIC_L_CAL
    if LCAL_NIC.is_file():
        nic = json.loads(LCAL_NIC.read_text(encoding="utf-8"))
        latency_nic = float(nic.get("adopted_for_o_recon_core50_nic_s") or NIC_L_CAL)
    print("hold", high_bytes, "nic_l_cal", latency_nic)
    src = (CFG / SRC_ORECON).read_text(encoding="utf-8")
    TASK.mkdir(parents=True, exist_ok=True)
    for sequence in ("low_high_low", "high_low_high"):
        text = retarget_h4(src, experiment_id="V4_g2_dev_h4_more_mem8g")
        text = to_nic_orecon(text, latency_nic)
        write_yaml(
            text,
            stem=f"g2_core50_nic_orecon_h4_{sequence}_seed17_mem8g",
            sequence=sequence,
            high_bytes=high_bytes,
            notes=(
                f"G2 H4 NIC O-recon {sequence}. L_cal={latency_nic}s from static NIC S0, not H4. "
                "Co-running pressure, not a freeze. Not 48-slot formal."
            ),
        )
    for sequence in ("low_high_low", "high_low_high"):
        text = retarget_h4(src, experiment_id="V4_g2_dev_h4_more_mem8g")
        text = to_br_adapt(text)
        write_yaml(
            text,
            stem=f"g2_core50_nc_br_adapt_h4_{sequence}_seed17_mem8g",
            sequence=sequence,
            high_bytes=high_bytes,
            notes=(
                f"G2 H4 NC BR-adapt {sequence}. Batch/replay adaptive only, plugins none. "
                "Not O-recon. Co-running pressure, not a freeze. Not 48-slot formal."
            ),
        )
    write_task("g2-occupancy-h4-more", ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"])
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "high_bytes": high_bytes,
        "nic_l_cal_s": latency_nic,
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "deferred": ["dynamic_sstar_needs_sequence_search"],
        "note": "NIC O-recon uses static NIC S0 L_cal. BR-adapt is H4 fourth method, not O-recon.",
    }
    (TASK / "_batch_g2_dev_h4_more.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(TASK_IDS), "high_bytes", high_bytes)


if __name__ == "__main__":
    main()
