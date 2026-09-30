"""G2 mem16g remaining-stream development batch. Not a freeze. kind=probe only.

Covers holes left after g2-dev-mem16g / mem16g-b:
NIC algorithms (not MAX-A: 79-exp batch=1 would be multi-day),
NI GSS/GEM/AGEM/MAX-A/MAX-P,
NC remaining methods on this capacity,
CIFAR10/100 MAX-A/MAX-P,
Endless A22 grouped methods (not official robot protocol).
"""

from __future__ import annotations

import json
import re
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    COMMON_TAIL,
    cifar10_dataset,
    cifar100_dataset,
    core50_dataset,
    core50_ni_dataset,
    core50_nic_dataset,
    endless_dataset,
    header,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "fullmem_v4"
TASK = ROOT / "experiments" / "fullmem_v4" / "tasks"
BATCH = "g2-dev-mem16g-c"
SEQ = 900
CAP = "mem16g"
EXP = "V4_g2_dev_mem16g_c"
GSS_EXTRA = "  mem_strength: 1\n  input_size:\n  - 3\n  - 32\n  - 32\n"
GEM20 = "  patterns_per_exp: 20\n  memory_strength: 0.5\n"
GEM50 = "  patterns_per_exp: 50\n  memory_strength: 0.5\n"
AGEM20 = "  patterns_per_exp: 20\n  sample_size: 64\n"
AGEM50 = "  patterns_per_exp: 50\n  sample_size: 64\n"
MAXA50 = "  patterns_per_exp: 50\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n"
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
    print("task", dest.name)


def stamp_mem16(text: str) -> str:
    text = text.replace("board_capacity_id: mem64g", f"board_capacity_id: {CAP}")
    text = text.replace("board_capacity_id: mem32g", f"board_capacity_id: {CAP}")
    text = text.replace("pressure_role: g2_dev_calibration_mem64g", "pressure_role: g2_dev_calibration_mem16g")
    text = text.replace("pressure_role: g2_dev_calibration_mem32g", "pressure_role: g2_dev_calibration_mem16g")
    return text


def clone(src_name: str, *, stem: str, run_id: str, notes: str) -> None:
    text = (CFG / src_name).read_text(encoding="utf-8")
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit(f"no run_id in {src_name}")
    text = text.replace(f"run_id: {old_run}", f"run_id: {run_id}", 1)
    text = re.sub(r"^experiment_id: .*$", f"experiment_id: {EXP}", text, count=1, flags=re.M)
    text = re.sub(r"^- V4_.*$", f"- {EXP}", text, count=1, flags=re.M)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("notes:"):
            lines[i] = f"notes: {json.dumps(notes)}"
    text = stamp_mem16("\n".join(lines) + "\n")
    dest = CFG / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)
    print("yaml", dest.name)
    write_task(stem.replace("_", "-"), ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"])


def emit_spec(spec: dict) -> None:
    start_enabled = spec.get("start_enabled", spec["plugins"] != "none")
    plugin_policy = spec.get(
        "plugin_policy",
        "fixed_advanced" if start_enabled else "fixed_default",
    )
    text = (
        header(
            spec["run_id"],
            EXP,
            spec["method_id"],
            spec["num_classes"],
            spec["plugins"],
            spec["new_batch"],
            spec["replay_batch"],
            start_enabled=start_enabled,
            algo_base=spec.get("algo_base", "er"),
            extra_algo=spec.get("extra_algo"),
            eval_batch=int(spec.get("eval_batch", 32)),
            seed=17,
        )
        + spec["dataset"]()
        + COMMON_TAIL.format(
            replay=spec["replay"],
            v3_dataset=spec["v3_dataset"],
            notes=json.dumps(spec["notes"]),
            plugin_policy=plugin_policy,
            prefetch_enabled=str(bool(spec.get("prefetch_enabled", False))).lower(),
            queue_depth=int(spec.get("queue_depth", 1)),
            replay_policy=spec.get("replay_policy", "experience_balanced"),
            replay_repr=spec.get("replay_repr", "avalanche_buffer"),
            controller_enabled=str(bool(spec.get("controller_enabled", False))).lower(),
            latency_th_s=spec.get("latency_th_s", 30.0),
            mb0=spec.get("mb0", 16.0),
            mr0=spec.get("mr0", 200.0),
            m_batch=spec.get("m_batch", 1.0),
            m_frame=spec.get("m_frame", 1.0),
            controlled_resource=spec.get("controlled_resource", "device"),
        )
    )
    text = stamp_mem16(text)
    dest = CFG / f"{spec['stem']}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    YAML_NAMES.append(dest.name)
    print("yaml", dest.name)
    write_task(spec["task_id"], ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{spec['stem']}.yaml"])


def gss(v3: str, dataset, nclass: int, stem: str, task_id: str, notes: str) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": "GSS",
        "num_classes": nclass,
        "plugins": "none",
        "algo_base": "gss",
        "extra_algo": GSS_EXTRA,
        "replay_policy": "gss_greedy",
        "replay_repr": "avalanche_gss",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def gem(v3: str, dataset, nclass: int, extra: str, stem: str, task_id: str, notes: str) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": "GEM",
        "num_classes": nclass,
        "plugins": "none",
        "algo_base": "gem",
        "extra_algo": extra,
        "replay_policy": "gem_patterns_per_exp",
        "replay_repr": "avalanche_gem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def agem(v3: str, dataset, nclass: int, extra: str, stem: str, task_id: str, notes: str) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": "AGEM",
        "num_classes": nclass,
        "plugins": "none",
        "algo_base": "agem",
        "extra_algo": extra,
        "replay_policy": "agem_patterns_per_exp",
        "replay_repr": "avalanche_agem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def plugin(
    method: str,
    plugins: str,
    v3: str,
    dataset,
    nclass: int,
    extra: str | None,
    stem: str,
    task_id: str,
    notes: str,
) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": method,
        "num_classes": nclass,
        "plugins": plugins,
        "extra_algo": extra,
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def maxa(v3: str, dataset, nclass: int, extra: str, stem: str, task_id: str, notes: str) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": "MAX-A_reconstructed",
        "num_classes": nclass,
        "plugins": "gem_ewc",
        "extra_algo": extra,
        "new_batch": 1,
        "replay_batch": 1,
        "replay": 200,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def maxp(v3: str, dataset, nclass: int, stem: str, task_id: str, notes: str) -> dict:
    return {
        "stem": stem,
        "run_id": f"fullmem_v4_{stem}",
        "method_id": "MAX-P_reconstructed",
        "num_classes": nclass,
        "plugins": "none",
        "new_batch": 256,
        "replay_batch": 10,
        "replay": 10,
        "mr0": 10.0,
        "dataset": dataset,
        "v3_dataset": v3,
        "notes": notes,
        "task_id": task_id,
    }


def main() -> None:
    TASK.mkdir(parents=True, exist_ok=True)
    nic = core50_nic_dataset
    ni = core50_ni_dataset
    generated = [
        gss("core50_nic", nic, 50, "g2_core50_nic_gss_seed17_mem16g", "g2-dev-core50-nic-gss-seed17-mem16g", "G2 mem16g NIC GSS seed=17. Not a freeze. Not official pressure."),
        gem("core50_nic", nic, 50, GEM50, "g2_core50_nic_gem_seed17_mem16g", "g2-dev-core50-nic-gem-seed17-mem16g", "G2 mem16g NIC GEM seed=17. Not a freeze."),
        agem("core50_nic", nic, 50, AGEM50, "g2_core50_nic_agem_seed17_mem16g", "g2-dev-core50-nic-agem-seed17-mem16g", "G2 mem16g NIC AGEM seed=17. Not a freeze."),
        plugin("ER+GEM", "gem", "core50_nic", nic, 50, GEM50, "g2_core50_nic_er_gem_seed17_mem16g", "g2-dev-core50-nic-er-gem-seed17-mem16g", "G2 mem16g NIC ER+GEM seed=17. Not a freeze."),
        plugin("ER+EWC", "ewc", "core50_nic", nic, 50, "  ewc_lambda: 100.0\n", "g2_core50_nic_er_ewc_seed17_mem16g", "g2-dev-core50-nic-er-ewc-seed17-mem16g", "G2 mem16g NIC ER+EWC seed=17. Not a freeze."),
        maxp("core50_nic", nic, 50, "g2_core50_nic_maxp_seed17_mem16g", "g2-dev-core50-nic-maxp-seed17-mem16g", "G2 mem16g NIC MAX-P_reconstructed seed=17. Not aliased from S*. NIC MAX-A deferred (batch=1 x 79 exp)."),
        gss("core50_ni", ni, 50, "g2_core50_ni_gss_seed17_mem16g", "g2-dev-core50-ni-gss-seed17-mem16g", "G2 mem16g NI GSS seed=17. Not a freeze."),
        gem("core50_ni", ni, 50, GEM50, "g2_core50_ni_gem_seed17_mem16g", "g2-dev-core50-ni-gem-seed17-mem16g", "G2 mem16g NI GEM seed=17. Not a freeze."),
        agem("core50_ni", ni, 50, AGEM50, "g2_core50_ni_agem_seed17_mem16g", "g2-dev-core50-ni-agem-seed17-mem16g", "G2 mem16g NI AGEM seed=17. Not a freeze."),
        maxa("core50_ni", ni, 50, MAXA50, "g2_core50_ni_maxa_seed17_mem16g", "g2-dev-core50-ni-maxa-seed17-mem16g", "G2 mem16g NI MAX-A_reconstructed A11. Not aliased from S*. Not a freeze."),
        maxp("core50_ni", ni, 50, "g2_core50_ni_maxp_seed17_mem16g", "g2-dev-core50-ni-maxp-seed17-mem16g", "G2 mem16g NI MAX-P_reconstructed. Not a freeze."),
    ]
    for stream, n_exp, scenario, v3 in (
        ("endless_ic", 4, "Classes", "endless_ic"),
        ("endless_il", 5, "Illumination", "endless_il"),
        ("endless_wc", 5, "Weather", "endless_wc"),
    ):
        ds = partial(endless_dataset, stream, n_exp, scenario)
        generated.extend(
            [
                gss(v3, ds, 5, f"g2_{stream}_gss_seed17_mem16g", f"g2-dev-{stream.replace('_', '-')}-gss-seed17-mem16g", f"G2 mem16g {stream} GSS seed=17. A22 grouped holdout, not official robot protocol."),
                gem(v3, ds, 5, GEM20, f"g2_{stream}_gem_seed17_mem16g", f"g2-dev-{stream.replace('_', '-')}-gem-seed17-mem16g", f"G2 mem16g {stream} GEM seed=17. A22 grouped holdout, not official robot protocol."),
                agem(v3, ds, 5, AGEM20, f"g2_{stream}_agem_seed17_mem16g", f"g2-dev-{stream.replace('_', '-')}-agem-seed17-mem16g", f"G2 mem16g {stream} AGEM seed=17. A22 grouped holdout, not official robot protocol."),
                plugin("ER+GEM+EWC", "gem_ewc", v3, ds, 5, None, f"g2_{stream}_er_gem_ewc_seed17_mem16g", f"g2-dev-{stream.replace('_', '-')}-er-gem-ewc-seed17-mem16g", f"G2 mem16g {stream} ER+GEM+EWC seed=17. A22 grouped holdout, not official robot protocol."),
                maxp(v3, ds, 5, f"g2_{stream}_maxp_seed17_mem16g", f"g2-dev-{stream.replace('_', '-')}-maxp-seed17-mem16g", f"G2 mem16g {stream} MAX-P_reconstructed. A22 grouped holdout, not official robot protocol."),
            ]
        )
    for spec in generated:
        emit_spec(spec)

    clones = [
        ("g2_cifar100_maxa_seed17.yaml", "g2_cifar100_maxa_seed17_mem16g", "fullmem_v4_g2_cifar100_maxa_seed17_mem16g", "G2 mem16g CIFAR100 MAX-A_reconstructed A11. Not aliased from S*. Not a freeze."),
        ("g2_cifar100_maxp_seed17.yaml", "g2_cifar100_maxp_seed17_mem16g", "fullmem_v4_g2_cifar100_maxp_seed17_mem16g", "G2 mem16g CIFAR100 MAX-P_reconstructed. Not a freeze."),
        ("g2_cifar10_maxa_seed17.yaml", "g2_cifar10_maxa_seed17_mem16g", "fullmem_v4_g2_cifar10_maxa_seed17_mem16g", "G2 mem16g CIFAR10 MAX-A_reconstructed A11. Not a freeze."),
        ("g2_cifar10_maxp_seed17.yaml", "g2_cifar10_maxp_seed17_mem16g", "fullmem_v4_g2_cifar10_maxp_seed17_mem16g", "G2 mem16g CIFAR10 MAX-P_reconstructed. Not a freeze."),
        ("g2_core50_nc_maxa_seed17.yaml", "g2_core50_nc_maxa_seed17_mem16g", "fullmem_v4_g2_core50_nc_maxa_seed17_mem16g", "G2 mem16g NC MAX-A_reconstructed A11. Not a freeze."),
        ("g2_core50_nc_maxp_seed17.yaml", "g2_core50_nc_maxp_seed17_mem16g", "fullmem_v4_g2_core50_nc_maxp_seed17_mem16g", "G2 mem16g NC MAX-P_reconstructed. Not a freeze."),
        ("g2_core50_nc_gem_seed17.yaml", "g2_core50_nc_gem_seed17_mem16g", "fullmem_v4_g2_core50_nc_gem_seed17_mem16g", "G2 mem16g NC GEM seed=17. Not a freeze."),
        ("g2_core50_nc_agem_seed17.yaml", "g2_core50_nc_agem_seed17_mem16g", "fullmem_v4_g2_core50_nc_agem_seed17_mem16g", "G2 mem16g NC AGEM seed=17. Not a freeze."),
        ("g2_core50_nc_er_gem_seed17.yaml", "g2_core50_nc_er_gem_seed17_mem16g", "fullmem_v4_g2_core50_nc_er_gem_seed17_mem16g", "G2 mem16g NC ER+GEM seed=17. Not a freeze."),
        ("g2_core50_nc_er_ewc_seed17.yaml", "g2_core50_nc_er_ewc_seed17_mem16g", "fullmem_v4_g2_core50_nc_er_ewc_seed17_mem16g", "G2 mem16g NC ER+EWC seed=17. Not a freeze."),
        ("g2_core50_nc_lr_seed17.yaml", "g2_core50_nc_lr_seed17_mem16g", "fullmem_v4_g2_core50_nc_lr_seed17_mem16g", "G2 mem16g NC LR reconstructed seed=17. Not a freeze."),
    ]
    for src, stem, run_id, notes in clones:
        clone(src, stem=stem, run_id=run_id, notes=notes)

    write_task("g2-occupancy-mem16g-c", ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"])
    write_task("g2-idle-mem16g-c", ["-m", "orion_repro.stages.fullmem_v4", "g2-idle-baseline"])
    manifest = {
        "batch_id": BATCH,
        "required_capacity": CAP,
        "kind": "probe",
        "not_a_freeze": True,
        "yamls": YAML_NAMES,
        "tasks": TASK_IDS,
        "deferred": ["NIC MAX-A_reconstructed: batch=1 x 79 experiences estimated multi-day; keep as cost gap"],
    }
    dest = TASK / "_batch_g2_dev_mem16g_c.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("n_tasks", len(TASK_IDS))


if __name__ == "__main__":
    main()
