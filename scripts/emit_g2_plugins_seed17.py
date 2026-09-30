"""Emit seed=17 ER+plugin attribution probes. Not a freeze.

PLAN C / H5b development: same stream as LR/S*/Oracle seed=17.
Does not overwrite seed=0 plugin calibration runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    COMMON_TAIL,
    ROOT,
    cifar100_dataset,
    header,
    write_task,
)


def render(spec: dict) -> str:
    start_enabled = spec.get("start_enabled", spec["plugins"] != "none")
    plugin_policy = spec.get(
        "plugin_policy",
        "fixed_advanced" if start_enabled else "fixed_default",
    )
    return (
        header(
            spec["run_id"],
            spec["experiment_id"],
            spec["method_id"],
            spec["num_classes"],
            spec["plugins"],
            spec["new_batch"],
            spec["replay_batch"],
            start_enabled=start_enabled,
            algo_base=spec.get("algo_base", "er"),
            extra_algo=spec.get("extra_algo"),
            eval_batch=int(spec.get("eval_batch", 32)),
            seed=int(spec.get("seeds", 17)),
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


SPECS = [
    {
        "stem": "g2_cifar100_er_gem_seed17",
        "run_id": "fullmem_v4_g2_cifar100_er_gem_seed17",
        "experiment_id": "V4_g2_dev_cifar100_plugins_seed17",
        "method_id": "ER+GEM",
        "num_classes": 100,
        "plugins": "gem",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 PLAN C plugin attribution: ER+GEM fixed on, seed=17. Does not overwrite seed=0 er_gem. Not a freeze.",
        "task_id": "g2-dev-cifar100-er-gem-seed17",
        "seq": 290,
    },
    {
        "stem": "g2_cifar100_er_ewc_seed17",
        "run_id": "fullmem_v4_g2_cifar100_er_ewc_seed17",
        "experiment_id": "V4_g2_dev_cifar100_plugins_seed17",
        "method_id": "ER+EWC",
        "num_classes": 100,
        "plugins": "ewc",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 PLAN C plugin attribution: ER+EWC fixed on, seed=17. Not a freeze.",
        "task_id": "g2-dev-cifar100-er-ewc-seed17",
        "seq": 291,
    },
    {
        "stem": "g2_cifar100_er_gem_ewc_seed17",
        "run_id": "fullmem_v4_g2_cifar100_er_gem_ewc_seed17",
        "experiment_id": "V4_g2_dev_cifar100_plugins_seed17",
        "method_id": "ER+GEM+EWC",
        "num_classes": 100,
        "plugins": "gem_ewc",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 PLAN C plugin attribution: ER+GEM+EWC fixed on, seed=17. Not a freeze.",
        "task_id": "g2-dev-cifar100-er-gem-ewc-seed17",
        "seq": 292,
    },
]


def main() -> None:
    cfg_dir = ROOT / "configs" / "fullmem_v4"
    for spec in SPECS:
        dest = cfg_dir / f"{spec['stem']}.yaml"
        dest.write_text(render(spec), encoding="utf-8", newline="\n")
        print("yaml", dest.name)
        write_task(
            spec["task_id"],
            spec["seq"],
            spec.get("kind", "probe"),
            ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{spec['stem']}.yaml"],
            "g2-dev-plugins-seed17",
        )
    write_task(
        "g2-occupancy-e",
        293,
        "probe",
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
        "g2-dev-plugins-seed17",
    )


if __name__ == "__main__":
    main()
