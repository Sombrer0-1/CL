"""Emit H6a preference and H6b threshold-decay development probes.

kind=probe. seed=17. mem16g. Not a freeze. Not kind=train.
H6a: balanced / latency / P-S / memory on the five H1 streams.
H6b: ln2 and N-1 half-lives on the same O-recon base; delta0 is H6a balanced.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    cifar10_dataset,
    cifar100_dataset,
    core50_dataset,
    core50_ni_dataset,
    core50_nic_dataset,
    header,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-h6-pref-delta"
SEQ = 960
CAP = "mem16g"
EXP = "V4_g2_dev_h6_mem16g"
M_MAX_MIB = 16435592 / 1024.0  # live MemTotal on current mem=16G boot
TASK_IDS: list[str] = []
YAML_NAMES: list[str] = []

H6A_ORDERS = {
    "balanced": None,
    "latency": ["latency", "memory", "plasticity", "stability"],
    "p_s": ["plasticity", "stability", "memory", "latency"],
    "memory": ["memory", "latency", "stability", "plasticity"],
}

STREAMS = [
    {
        "key": "cifar10",
        "run_key": "cifar10",
        "num_classes": 10,
        "n_exp": 10,
        "latency_th_s": 5.632,
        "v3_dataset": "splitcifar10",
        "dataset": cifar10_dataset,
        "extra_algo": None,
    },
    {
        "key": "cifar100",
        "run_key": "cifar100",
        "num_classes": 100,
        "n_exp": 10,
        "latency_th_s": 5.513,
        "v3_dataset": "splitcifar100",
        "dataset": cifar100_dataset,
        "extra_algo": None,
    },
    {
        "key": "core50-ni",
        "run_key": "core50_ni",
        "num_classes": 50,
        "n_exp": 8,
        "latency_th_s": 19.308,
        "v3_dataset": "core50_ni",
        "dataset": core50_ni_dataset,
        "extra_algo": None,
    },
    {
        "key": "core50-nc",
        "run_key": "core50_nc",
        "num_classes": 50,
        "n_exp": 9,
        "latency_th_s": 15.706,
        "v3_dataset": "core50_nc",
        "dataset": core50_dataset,
        "extra_algo": "  patterns_per_exp: 50\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n",
    },
    {
        "key": "core50-nic",
        "run_key": "core50_nic",
        "num_classes": 50,
        "n_exp": 79,
        "latency_th_s": 1.871,
        "v3_dataset": "core50_nic",
        "dataset": core50_nic_dataset,
        "extra_algo": "  patterns_per_exp: 50\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n",
    },
]


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def h6_delta(kind: str, n_exp: int) -> float:
    if kind == "delta0":
        return 0.0
    if kind == "ln2":
        return math.log(2.0)
    if kind == "n-1":
        return math.log(2.0) / float(n_exp - 1)
    raise ValueError(kind)


def preference_block(name: str) -> str:
    order = H6A_ORDERS[name]
    if order is None:
        return "  preference: balanced\n"
    lines = [f"  preference: {name}", "  preference_order:"]
    lines.extend(f"  - {item}" for item in order)
    return "\n".join(lines) + "\n"


def render(stream: dict, *, variant: str, preference: str, delta_kind: str, notes: str) -> str:
    run_id = f"fullmem_v4_g2_{stream['run_key']}_{variant}_seed17_{CAP}"
    text = (
        header(
            run_id,
            EXP,
            "O-recon",
            stream["num_classes"],
            "gem_ewc",
            16,
            16,
            start_enabled=False,
            extra_algo=stream["extra_algo"],
            seed=17,
        )
        + stream["dataset"]()
        + f"""replay:
  capacity: 200
  policy: experience_balanced
  representation: avalanche_buffer
controller:
  enabled: true
  units:
    accuracy: ratio
    latency: seconds
    memory: MiB
  metric_definition: diag_initial
  feedback_source: development_val_seen
{preference_block(preference)}  coefficients:
    kp: 0.25
    ks: 0.25
    kl: 0.25
    km: 0.25
  thresholds:
    p: 0.5
    s: 0.5
    latency_s: {stream['latency_th_s']}
    m_max_mib: {M_MAX_MIB}
  thr0: 0.05
  delta: {h6_delta(delta_kind, stream['n_exp'])}
  updates:
    alpha: 0.1
    beta: 0.2
  mb0: 49152.0
  mr0: 614400.0
  m_batch: 3072.0
  m_frame: 3072.0
  min_batch: 1
  max_batch: 1024
  equal_uses_gt: true
  plugin_policy: adaptive
budget:
  enforcement: observed_only
  controlled_resource: board
  limit_bytes: null
  guard_mode: stop
  cost_model_path: null
prefetch:
  enabled: true
  queue_depth: 2
  pin_memory: false
  num_workers: 0
measurement:
  sample_interval_ms: 100
  checkpoint_policy: none
study_id: fullmem_v4
reuse_version: 4
pressure_role: g2_dev_h6_mem16g
v3_role: not_v3
v3_dataset: {stream['v3_dataset']}
board_capacity_id: {CAP}
notes: {json.dumps(notes)}
"""
    )
    return text


def write_task(task_id: str, config_rel: str) -> None:
    payload = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": task_id,
        "seq": next_seq(),
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.run",
            "--config",
            config_rel,
        ],
        "required_capacity": CAP,
        "batch_id": BATCH,
    }
    dest = TASK / f"{task_id}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    TASK_IDS.append(task_id)
    print("task", dest.name)


def emit_one(stream: dict, *, variant: str, preference: str, delta_kind: str, notes: str) -> None:
    yaml_name = f"g2_{stream['run_key']}_{variant}_seed17_{CAP}.yaml"
    text = render(stream, variant=variant, preference=preference, delta_kind=delta_kind, notes=notes)
    (CFG / yaml_name).write_text(text, encoding="utf-8", newline="\n")
    YAML_NAMES.append(yaml_name)
    print("yaml", yaml_name)
    task_id = f"g2-dev-{stream['key']}-{variant}-seed17-{CAP}"
    write_task(task_id, f"configs/fullmem_v4/{yaml_name}")


def main() -> None:
    CFG.mkdir(parents=True, exist_ok=True)
    TASK.mkdir(parents=True, exist_ok=True)
    for stream in STREAMS:
        n = stream["n_exp"]
        for pref in ("balanced", "latency", "p_s", "memory"):
            emit_one(
                stream,
                variant=f"h6a_{pref}",
                preference=pref,
                delta_kind="delta0",
                notes=(
                    f"H6a development: O-recon preference={pref} on {stream['key']}, "
                    f"seed=17, mem16g, L_cal={stream['latency_th_s']}s, M_max=live MemTotal. "
                    "Not a freeze. Factor/action may not split when factor_m saturates."
                ),
            )
        emit_one(
            stream,
            variant="h6b_ln2",
            preference="balanced",
            delta_kind="ln2",
            notes=(
                f"H6b development: δ=ln2 (half-life 1 experience) on {stream['key']}, "
                f"N={n}, seed=17, mem16g. Not a freeze."
            ),
        )
        emit_one(
            stream,
            variant="h6b_nminus1",
            preference="balanced",
            delta_kind="n-1",
            notes=(
                f"H6b development: δ=ln2/(N-1) (half-life spans stream) on {stream['key']}, "
                f"N={n}, seed=17, mem16g. Not literal δ=N-1. Not a freeze."
            ),
        )
    occ = {
        "study_id": "fullmem_v4",
        "design_version": "design-v1",
        "task_id": "g2-occupancy-h6",
        "seq": next_seq(),
        "kind": "probe",
        "argv": [
            "/home/zhuzetong/miniconda3/envs/orion/bin/python",
            "-m",
            "orion_repro.stages.fullmem_v4",
            "g2-occupancy",
        ],
        "required_capacity": CAP,
        "batch_id": BATCH,
    }
    (TASK / "g2-occupancy-h6.json").write_text(json.dumps(occ, indent=2) + "\n", encoding="utf-8", newline="\n")
    TASK_IDS.append("g2-occupancy-h6")
    print("task g2-occupancy-h6.json")
    man = {
        "batch_id": BATCH,
        "kind": "probe",
        "required_capacity": CAP,
        "not_a_freeze": True,
        "n_tasks": len(TASK_IDS),
        "tasks": TASK_IDS,
        "yamls": YAML_NAMES,
        "note": "H6a 4 prefs x 5 streams + H6b ln2/N-1 x 5 streams + occupancy. seed=17 mem16g.",
    }
    (TASK / "_batch_g2_dev_h6.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("batch", BATCH, "n", len(TASK_IDS))


if __name__ == "__main__":
    main()
