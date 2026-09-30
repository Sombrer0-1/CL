"""Emit G2 LR reconstructed development probes, seed=17. Not a freeze."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    COMMON_TAIL,
    ROOT,
    cifar100_dataset,
    core50_dataset,
    header,
    write_task,
)


def render(spec: dict) -> str:
    return (
        header(
            spec["run_id"],
            spec["experiment_id"],
            spec["method_id"],
            spec["num_classes"],
            spec["plugins"],
            spec["new_batch"],
            spec["replay_batch"],
            start_enabled=False,
            algo_base="lr",
            extra_algo=spec.get("extra_algo"),
            eval_batch=32,
            seed=17,
        )
        + spec["dataset"]()
        + COMMON_TAIL.format(
            replay=spec["replay"],
            v3_dataset=spec["v3_dataset"],
            notes=json.dumps(spec["notes"]),
            plugin_policy="fixed_default",
            prefetch_enabled="false",
            queue_depth=1,
            replay_policy="random_latent",
            replay_repr="layer2_float32",
            controller_enabled="false",
            latency_th_s=30.0,
            mb0=16.0,
            mr0=200.0,
            m_batch=1.0,
            m_frame=1.0,
            controlled_resource="device",
        )
    )


SPECS = [
    {
        "stem": "g2_cifar100_lr_seed17",
        "run_id": "fullmem_v4_g2_cifar100_lr_seed17",
        "experiment_id": "V4_g2_dev_cifar100_lr_seed17",
        "method_id": "LR_reconstructed",
        "num_classes": 100,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": cifar100_dataset,
        "v3_dataset": "splitcifar100",
        "notes": "G2 LR reconstructed (A11): layer2 latents, freeze stem after exp0, seed=17. Not author Orion. Not a freeze. MAX-A/MAX-P remain unwired.",
        "task_id": "g2-dev-cifar100-lr-seed17",
        "seq": 280,
    },
    {
        "stem": "g2_core50_nc_lr_seed17",
        "run_id": "fullmem_v4_g2_core50_nc_lr_seed17",
        "experiment_id": "V4_g2_dev_core50_nc_lr_seed17",
        "method_id": "LR_reconstructed",
        "num_classes": 50,
        "plugins": "none",
        "new_batch": 16,
        "replay_batch": 16,
        "replay": 200,
        "dataset": core50_dataset,
        "v3_dataset": "core50_nc",
        "notes": "G2 LR reconstructed on CORe50-NC, seed=17. Same A11 semantics. Not a freeze.",
        "task_id": "g2-dev-core50-nc-lr-seed17",
        "seq": 281,
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
            "g2-dev-lr",
        )


if __name__ == "__main__":
    main()
