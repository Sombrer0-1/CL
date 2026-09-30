"""Emit seed=17 S* six-cell and Oracle 42-cell development searches.

Does not rewrite seed=0 exploration configs. Selection is not a freeze.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    COMMON_TAIL,
    ORACLE_BATCHES,
    ORACLE_REPLAY,
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


def sstar_specs() -> list[dict]:
    specs = []
    seq = 210
    for batch in (16, 64, 256):
        for replay in (200, 2000):
            cell = f"b{batch}_r{replay}"
            stem = f"g2_cifar100_sstar_{cell}_seed17"
            specs.append(
                {
                    "stem": stem,
                    "run_id": f"fullmem_v4_g2_cifar100_sstar_{cell}_seed17",
                    "experiment_id": "V4_g2_dev_cifar100_sstar_seed17",
                    "method_id": "Sstar_candidate",
                    "num_classes": 100,
                    "plugins": "none",
                    "new_batch": batch,
                    "replay_batch": batch,
                    "replay": replay,
                    "seeds": 17,
                    "dataset": cifar100_dataset,
                    "v3_dataset": "splitcifar100",
                    "notes": (
                        f"G2 S* six-cell development search seed=17, cell {cell}. "
                        "replay_batch equals new_batch as in dedicated S* remaining cells; "
                        "plugins/storage fixed. Seed-0 exploration cannot freeze. Not a freeze."
                    ),
                    "task_id": f"g2-dev-cifar100-sstar-{cell.replace('_', '-')}-seed17",
                    "seq": seq,
                    "kind": "probe",
                    "argv_extra": ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
                }
            )
            seq += 1
    return specs


def oracle_seed17_specs() -> list[dict]:
    specs = []
    seq = 220
    for batch in ORACLE_BATCHES:
        for replay in ORACLE_REPLAY:
            cell = f"b{batch}_r{replay}"
            stem = f"g2_oracle_cifar100_{cell}_seed17"
            specs.append(
                {
                    "stem": stem,
                    "run_id": f"fullmem_v4_g2_oracle_cifar100_{cell}_seed17",
                    "experiment_id": "V4_g2_oracle_cifar100_seed17",
                    "method_id": "oracle_reconstructed",
                    "num_classes": 100,
                    "plugins": "none",
                    "new_batch": batch,
                    "replay_batch": batch,
                    "replay": replay,
                    "seeds": 17,
                    "dataset": cifar100_dataset,
                    "v3_dataset": "splitcifar100",
                    "notes": (
                        f"G2 Oracle 42-cell search on CIFAR100 development stream seed=17, cell {cell}. "
                        "Seed-0 exploration cannot freeze. Search cost not a selected-config train cost."
                    ),
                    "task_id": f"g2-oracle-cifar100-{cell.replace('_', '-')}-seed17",
                    "seq": seq,
                    "kind": "probe",
                    "argv_extra": ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
                }
            )
            seq += 1
    return specs


def main() -> None:
    cfg_dir = ROOT / "configs" / "fullmem_v4"
    specs = [*sstar_specs(), *oracle_seed17_specs()]
    for spec in specs:
        dest = cfg_dir / f"{spec['stem']}.yaml"
        dest.write_text(render(spec), encoding="utf-8", newline="\n")
        print("yaml", dest.name)
        write_task(
            spec["task_id"],
            spec["seq"],
            spec["kind"],
            spec["argv_extra"],
            "g2-dev-seed17-select",
        )
    print("emitted", len(specs))


if __name__ == "__main__":
    main()
