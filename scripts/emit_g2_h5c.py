"""Emit H5c scripted pause-keep-state vs release-rebuild probes. kind=probe. Not a freeze."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs" / "fullmem_v4" / "g2_cifar100_er_gem_ewc_seed17_mem16g.yaml"
BATCH = "g2-dev-h5c-plugin-lifecycle"
SEQ = 950
SCHEDULE = [
    "advanced",
    "advanced",
    "advanced",
    "default",
    "default",
    "advanced",
    "advanced",
    "default",
    "default",
    "default",
]


def _schedule_yaml() -> str:
    lines = ["  plugin_schedule:"]
    for item in SCHEDULE:
        lines.append(f"  - {item}")
    return "\n".join(lines) + "\n"


def _patch(text: str, *, run_id: str, method_id: str, policy: str, notes: str) -> str:
    text = text.replace(
        "run_id: fullmem_v4_g2_cifar100_er_gem_ewc_seed17_mem16g",
        f"run_id: {run_id}",
        1,
    )
    text = text.replace("method_id: ER+GEM+EWC", f"method_id: {method_id}", 1)
    text = text.replace("plugin_policy: fixed_advanced", f"plugin_policy: {policy}\n{_schedule_yaml().rstrip()}", 1)
    text = text.replace("pressure_role: g2_dev_calibration_mem64g", "pressure_role: g2_dev_h5c_mem16g", 1)
    text = text.replace(
        'notes: "G2 mem16g CIFAR100 ER+GEM+EWC seed=17. Not a freeze."',
        f"notes: {json.dumps(notes)}",
        1,
    )
    return text


def write_task(task_id: str, seq: int, config_rel: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": seq,
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.run",
            "--config",
            config_rel,
        ],
        "required_capacity": "mem16g",
        "batch_id": BATCH,
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("task", dest.name)


def main() -> None:
    pause = _patch(
        SRC.read_text(encoding="utf-8"),
        run_id="fullmem_v4_g2_cifar100_h5c_pause_keep_seed17_mem16g",
        method_id="O-eng-scripted-pause-keep",
        policy="scripted_pause_keep_state",
        notes="H5c diagnostic: scripted GEM+EWC pause-keep-state. Not O-recon. Not a freeze.",
    )
    release = _patch(
        SRC.read_text(encoding="utf-8"),
        run_id="fullmem_v4_g2_cifar100_h5c_release_rebuild_seed17_mem16g",
        method_id="O-eng-scripted-release-rebuild",
        policy="scripted_release_rebuild",
        notes="H5c diagnostic: scripted GEM+EWC release-rebuild. Not O-recon. Not a freeze.",
    )
    pause_name = "g2_cifar100_h5c_pause_keep_seed17_mem16g.yaml"
    release_name = "g2_cifar100_h5c_release_rebuild_seed17_mem16g.yaml"
    (ROOT / "configs" / "fullmem_v4" / pause_name).write_text(pause, encoding="utf-8", newline="\n")
    (ROOT / "configs" / "fullmem_v4" / release_name).write_text(release, encoding="utf-8", newline="\n")
    print("yaml", pause_name)
    print("yaml", release_name)
    write_task("g2-dev-cifar100-h5c-pause-keep-seed17-mem16g", SEQ, f"configs/fullmem_v4/{pause_name}")
    write_task("g2-dev-cifar100-h5c-release-rebuild-seed17-mem16g", SEQ + 1, f"configs/fullmem_v4/{release_name}")
    occ = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": "g2-occupancy-h5c",
        "seq": SEQ + 2,
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.stages.fullmem_v4",
            "g2-occupancy",
        ],
        "required_capacity": "mem16g",
        "batch_id": BATCH,
    }
    (ROOT / "experiments" / "fullmem_v4" / "tasks" / "g2-occupancy-h5c.json").write_text(
        json.dumps(occ, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("task g2-occupancy-h5c.json")
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": "mem16g",
        "tasks": [
            "g2-dev-cifar100-h5c-pause-keep-seed17-mem16g",
            "g2-dev-cifar100-h5c-release-rebuild-seed17-mem16g",
            "g2-occupancy-h5c",
        ],
        "yamls": [pause_name, release_name],
    }
    dest = ROOT / "experiments" / "fullmem_v4" / "tasks" / "_batch_g2_dev_h5c.json"
    dest.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("manifest", dest.name)


if __name__ == "__main__":
    main()
