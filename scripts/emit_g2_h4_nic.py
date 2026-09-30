"""Emit H4 NIC development probes. kind=probe. Not a freeze. Not 48-slot formal."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
CAL = ROOT / "reports" / "fullmem_v4" / "g2" / "h4_calibrate.json"
BATCH = "g2-dev-h4-nic"
SEQ = 1200
CAP = "mem8g"
TASK_IDS: list[str] = []
YAML_NAMES: list[str] = []
CHUNK = 256 * 1024 * 1024


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


def training_high_bytes(cal: dict) -> int:
    calibrated = int(cal.get("high_bytes") or 0)
    if calibrated <= 0:
        raise SystemExit(f"refuse emit: calibration high_bytes={calibrated}")
    if not cal.get("identifiable") and calibrated < CHUNK:
        raise SystemExit("refuse emit: hold is not identifiable")
    baseline = int(cal.get("baseline_mem_available_bytes") or 0)
    os_floor = 1024 * 1024 * 1024
    train_headroom = 1536 * 1024 * 1024
    training_cap = max(0, baseline - os_floor - train_headroom) if baseline else calibrated
    high_bytes = min(calibrated, training_cap)
    high_bytes = (high_bytes // CHUNK) * CHUNK
    if high_bytes < 1024 * 1024 * 1024:
        raise SystemExit(f"refuse emit: training hold {high_bytes} < 1GiB identifiability")
    return high_bytes


def emit_yaml(src_name: str, *, stem: str, run_id: str, task_id: str, sequence: str, high_bytes: int, notes: str) -> None:
    text = (CFG / src_name).read_text(encoding="utf-8")
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit(f"no run_id in {src_name}")
    text = text.replace(f"run_id: {old_run}", f"run_id: {run_id}", 1)
    text = re.sub(r"^experiment_id: .*$", "experiment_id: V4_g2_dev_h4_nic_mem8g", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", "- V4_g2_dev_h4_nic_mem8g", text, count=1, flags=re.M)
    for old in ("mem64g", "mem32g", "mem16g"):
        text = text.replace(f"board_capacity_id: {old}", f"board_capacity_id: {CAP}")
        text = text.replace(f"pressure_role: g2_dev_calibration_{old}", "pressure_role: g2_dev_h4_nic_mem8g")
    text = text.replace("pressure_role: g2_dev_calibration_mem8g", "pressure_role: g2_dev_h4_nic_mem8g")
    text = inject_envelope(text, sequence=sequence, high_bytes=high_bytes, notes=notes)
    dest = CFG / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)
    print("yaml", dest.name)
    write_task(task_id, ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{dest.name}"])


def main() -> None:
    cal_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CAL
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    high_bytes = training_high_bytes(cal)
    print("hold training_high_bytes", high_bytes, "calibrated", cal.get("high_bytes"))
    TASK.mkdir(parents=True, exist_ok=True)
    src = "g2_core50_nic_s0_seed17_mem16g.yaml"
    if not (CFG / src).is_file():
        raise SystemExit(f"missing {src}")
    for sequence in ("low_high_low", "high_low_high"):
        stem = f"g2_core50_nic_s0_h4_{sequence}_seed17_mem8g"
        emit_yaml(
            src,
            stem=stem,
            run_id=f"fullmem_v4_{stem}",
            task_id=f"g2-dev-core50-nic-s0-h4-{sequence.replace('_', '-')}-seed17-mem8g",
            sequence=sequence,
            high_bytes=high_bytes,
            notes=(
                f"G2 H4 NIC S0 {sequence} background hold. Co-running pressure, not static tight. "
                "Not a freeze. Not 48-slot formal. NIC O-recon deferred until NIC L_cal."
            ),
        )
    write_task("g2-occupancy-h4-nic", ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"])
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "high_bytes": high_bytes,
        "calibrated_high_bytes": cal.get("high_bytes"),
        "identifiable": cal.get("identifiable"),
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "deferred": ["nic_orecon_needs_stream_lcal", "dynamic_sstar_needs_sequence_search"],
        "note": "H4 NIC S0 both sequences. kind=probe only. Occupancy task_id is new so it is not idempotent with g2-occupancy-h4.",
    }
    (TASK / "_batch_g2_dev_h4_nic.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(TASK_IDS), "high_bytes", high_bytes)


if __name__ == "__main__":
    main()
