"""CIFAR-10 and CORe50-NI static S0 / S* / O-recon at mem8g.

L_cal is the median learning_s of the existing S0 runs:
CIFAR-10 5.632 s from fullmem_v4_g2_cifar10_s0,
NI 19.308 s from fullmem_v4_g2_core50_ni_s0.
Byte-space constants match the 32x32 O-recon already used on CIFAR-100.
GEM pattern count 20 matches the existing CIFAR-10 and NI ER+GEM+EWC configs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/zhuzetong/research/cl/Reproduce-Orion")
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g4-static-loose-mem8g"
SEQ = 4100

STREAMS = (
    ("cifar10", "cifar10", "g2_cifar10_s0_seed17_mem16g.yaml", 5.632),
    ("core50_ni", "core50-ni", "g2_core50_ni_s0_seed17_mem16g.yaml", 19.308),
)


def retarget(text: str, *, run_id: str, method_id: str, seed: int, notes: str) -> str:
    text = re.sub(r"^run_id: .*$", f"run_id: {run_id}", text, count=1, flags=re.M)
    text = re.sub(r"^phase: .*$", "phase: formal", text, count=1, flags=re.M)
    text = re.sub(r"^experiment_id: .*$", "experiment_id: V4_g4_static_loose_mem8g", text, count=1, flags=re.M)
    text = re.sub(r"^protocol_id: .*$", "protocol_id: fullmem_v4_g4_static_loose", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", "- V4_g4_static_loose_mem8g", text, count=1, flags=re.M)
    text = re.sub(r"^method_id: .*$", f"method_id: {method_id}", text, count=1, flags=re.M)
    text = re.sub(r"^  model: .*$", f"  model: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  stream: .*$", f"  stream: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  replay: .*$", f"  replay: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^  augmentation: .*$", f"  augmentation: {seed}", text, count=1, flags=re.M)
    text = text.replace("feedback_source: development_val_seen", "feedback_source: official_test_seen")
    text = text.replace("board_capacity_id: mem16g", "board_capacity_id: mem8g")
    text = text.replace("pressure_role: g2_dev_calibration_mem64g", "pressure_role: g4_static_loose_mem8g")
    text = re.sub(r"^notes:.*$", "notes: " + json.dumps(notes), text, count=1, flags=re.M)
    if "official_test_seen" not in text:
        raise SystemExit(f"{run_id} missing official feedback")
    if "resource_envelope:" in text:
        raise SystemExit(f"{run_id} has an envelope")
    return text


def as_orecon(text: str, latency_s: float) -> str:
    text = text.replace("optional_plugins: none", "optional_plugins: gem_ewc")
    text = text.replace(
        "  optional_start_enabled: false\n",
        "  optional_start_enabled: false\n  patterns_per_exp: 20\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n",
        1,
    )
    text = text.replace("  enabled: false\n  units:", "  enabled: true\n  units:", 1)
    text = text.replace("plugin_policy: fixed_default", "plugin_policy: adaptive")
    text = text.replace("latency_s: 30.0", f"latency_s: {latency_s}")
    text = text.replace("mb0: 16.0", "mb0: 49152.0")
    text = text.replace("mr0: 200.0", "mr0: 614400.0")
    text = text.replace("m_batch: 1.0", "m_batch: 3072.0")
    text = text.replace("m_frame: 1.0", "m_frame: 3072.0")
    text = text.replace("controlled_resource: device", "controlled_resource: board")
    text = text.replace("prefetch:\n  enabled: false", "prefetch:\n  enabled: true")
    if "plugin_policy: adaptive" not in text or "optional_plugins: gem_ewc" not in text:
        raise SystemExit("orecon rewrite failed")
    if f"latency_s: {latency_s}" not in text:
        raise SystemExit("latency rewrite failed")
    return text


def main() -> None:
    global SEQ
    freeze = json.loads((ROOT / "reports/fullmem_v4/g3/FREEZE.json").read_text())
    if BATCH not in (freeze.get("admitted_batches") or []):
        raise SystemExit("batch not admitted")
    if freeze.get("capacity_id") != "mem8g":
        raise SystemExit("capacity is not mem8g")
    tasks = []
    for dataset, task_ds, src_name, latency in STREAMS:
        base = (CFG / src_name).read_text()
        for seed in (0, 1, 2):
            for method, method_id, replay, orecon in (
                ("s0", "S0", False, False),
                ("sstar", "Sstar", True, False),
                ("orecon", "O-recon", False, True),
            ):
                text = as_orecon(base, latency) if orecon else base
                if replay:
                    if "capacity: 200\n" not in text:
                        raise SystemExit(f"{src_name} missing capacity 200")
                    text = text.replace("capacity: 200\n", "capacity: 2000\n", 1)
                stem = f"g4_{dataset}_{method}_static_seed{seed}_mem8g"
                notes = (
                    f"Formal static loose mem8g {dataset} {method_id} seed {seed}. "
                    f"No H4 holder. S* is development b16/r2000. "
                    f"O-recon L_cal={latency} from the existing S0 median learning_s."
                )
                text = retarget(
                    text,
                    run_id=f"fullmem_v4_{stem}",
                    method_id=method_id,
                    seed=seed,
                    notes=notes,
                )
                (CFG / f"{stem}.yaml").write_text(text)
                SEQ += 1
                task_id = f"g4-{task_ds}-{method}-static-seed{seed}-mem8g"
                payload = {
                    "study_id": "fullmem_v4",
                    "design_version": "design-v1",
                    "task_id": task_id,
                    "seq": SEQ,
                    "kind": "train",
                    "argv": [
                        "/home/zhuzetong/miniconda3/envs/orion/bin/python",
                        "-m",
                        "orion_repro.run",
                        "--config",
                        f"configs/fullmem_v4/{stem}.yaml",
                    ],
                    "required_capacity": "mem8g",
                    "batch_id": BATCH,
                }
                (TASK / f"{task_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
                tasks.append(task_id)
                print(task_id)
    (TASK / "_batch_g4_static_loose_rest_mem8g.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")
    print("n", len(tasks))


if __name__ == "__main__":
    main()
