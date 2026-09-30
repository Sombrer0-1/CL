"""Emit one large seed=17 G2 development queue. Not a freeze. Not kind=train.

Does not overwrite seed=0 runs or the in-flight CIFAR100 plugin seed=17 batch.
MAX-A/MAX-P use A11 reconstruction names, not S* aliases.
Avalanche 0.6.0 is the installed library for the missing paper numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_g2_configs import (  # noqa: E402
    COMMON_TAIL,
    ROOT,
    cifar10_dataset,
    cifar100_dataset,
    core50_dataset,
    core50_ni_dataset,
    core50_nic_dataset,
    endless_dataset,
    header,
    write_task,
)

BATCH = "g2-dev-seed17-wide"
CFG = ROOT / "configs" / "fullmem_v4"

GSS_EXTRA = "  mem_strength: 1\n  input_size:\n  - 3\n  - 32\n  - 32\n"
GEM20 = "  patterns_per_exp: 20\n  memory_strength: 0.5\n"
GEM50 = "  patterns_per_exp: 50\n  memory_strength: 0.5\n"
AGEM20 = "  patterns_per_exp: 20\n  sample_size: 64\n"
AGEM50 = "  patterns_per_exp: 50\n  sample_size: 64\n"
MAXA_EXTRA = "  patterns_per_exp: 50\n  memory_strength: 0.5\n  ewc_lambda: 100.0\n"
EWC_EXTRA = "  ewc_lambda: 100.0\n"
GEM_EWC20 = GEM20 + EWC_EXTRA

SEQ = 300


def next_seq() -> int:
    global SEQ
    SEQ += 1
    return SEQ - 1


def render(spec: dict) -> str:
    start_enabled = spec.get("start_enabled", spec.get("plugins", "none") != "none")
    plugin_policy = spec.get(
        "plugin_policy",
        "fixed_advanced" if start_enabled else "fixed_default",
    )
    algo_base = spec.get("algo_base", "er")
    replay_policy = spec.get("replay_policy", "experience_balanced")
    replay_repr = spec.get("replay_repr", "avalanche_buffer")
    if algo_base == "lr":
        start_enabled = False
        plugin_policy = "fixed_default"
        replay_policy = "random_latent"
        replay_repr = "layer2_float32"
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
            algo_base=algo_base,
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
            replay_policy=replay_policy,
            replay_repr=replay_repr,
            controller_enabled=str(bool(spec.get("controller_enabled", False))).lower(),
            latency_th_s=spec.get("latency_th_s", 30.0),
            mb0=spec.get("mb0", 16.0),
            mr0=spec.get("mr0", float(spec["replay"])),
            m_batch=spec.get("m_batch", 1.0),
            m_frame=spec.get("m_frame", 1.0),
            controlled_resource=spec.get("controlled_resource", "device"),
        )
    )


def item(**kwargs) -> dict:
    kwargs.setdefault("seq", next_seq())
    kwargs.setdefault("new_batch", 16)
    kwargs.setdefault("replay_batch", 16)
    kwargs.setdefault("replay", 200)
    kwargs.setdefault("plugins", "none")
    kwargs.setdefault("algo_base", "er")
    return kwargs


def stream(prefix: str, num_classes: int, dataset, v3: str, *, gem_pat: str, agem_pat: str) -> list[dict]:
    """S0 / LR / GSS / GEM / AGEM / ER plugins / MAX-P for one stream."""
    out = []
    out.append(
        item(
            stem=f"g2_{prefix}_s0_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_s0_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="S0",
            num_classes=num_classes,
            dataset=dataset,
            v3_dataset=v3,
            notes=f"G2 wide seed=17 S0 on {prefix}. Does not overwrite seed=0. Not a freeze.",
            task_id=f"g2-dev-{prefix.replace('_', '-')}-s0-seed17",
        )
    )
    out.append(
        item(
            stem=f"g2_{prefix}_lr_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_lr_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="LR_reconstructed",
            num_classes=num_classes,
            algo_base="lr",
            dataset=dataset,
            v3_dataset=v3,
            notes=f"G2 wide seed=17 LR reconstructed (A11) on {prefix}. Not author Orion.",
            task_id=f"g2-dev-{prefix.replace('_', '-')}-lr-seed17",
        )
    )
    out.append(
        item(
            stem=f"g2_{prefix}_gss_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_gss_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="GSS",
            num_classes=num_classes,
            algo_base="gss",
            extra_algo=GSS_EXTRA,
            replay_policy="gss_greedy",
            replay_repr="avalanche_gss",
            dataset=dataset,
            v3_dataset=v3,
            notes=f"G2 wide seed=17 GSS base on {prefix}. CIFAR100 seed=0 GSS had P=0.01; recheck. Not a freeze.",
            task_id=f"g2-dev-{prefix.replace('_', '-')}-gss-seed17",
        )
    )
    out.append(
        item(
            stem=f"g2_{prefix}_gem_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_gem_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="GEM",
            num_classes=num_classes,
            algo_base="gem",
            extra_algo=gem_pat,
            replay_policy="gem_patterns_per_exp",
            replay_repr="avalanche_gem",
            dataset=dataset,
            v3_dataset=v3,
            notes=f"G2 wide seed=17 GEM as base on {prefix}. Not an optional plugin.",
            task_id=f"g2-dev-{prefix.replace('_', '-')}-gem-seed17",
        )
    )
    out.append(
        item(
            stem=f"g2_{prefix}_agem_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_agem_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="AGEM",
            num_classes=num_classes,
            algo_base="agem",
            extra_algo=agem_pat,
            replay_policy="agem_patterns_per_exp",
            replay_repr="avalanche_agem",
            dataset=dataset,
            v3_dataset=v3,
            notes=f"G2 wide seed=17 AGEM as base on {prefix}. Not a freeze.",
            task_id=f"g2-dev-{prefix.replace('_', '-')}-agem-seed17",
        )
    )
    for plug, method, extra in (
        ("gem", "ER+GEM", gem_pat),
        ("ewc", "ER+EWC", EWC_EXTRA),
        ("gem_ewc", "ER+GEM+EWC", gem_pat + EWC_EXTRA),
    ):
        out.append(
            item(
                stem=f"g2_{prefix}_er_{plug}_seed17",
                run_id=f"fullmem_v4_g2_{prefix}_er_{plug}_seed17",
                experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
                method_id=method,
                num_classes=num_classes,
                plugins=plug,
                extra_algo=extra,
                dataset=dataset,
                v3_dataset=v3,
                notes=f"G2 wide PLAN C {method} fixed on, seed=17, {prefix}. Not a freeze.",
                task_id=f"g2-dev-{prefix.replace('_', '-')}-er-{plug.replace('_', '-')}-seed17",
            )
        )
    out.append(
        item(
            stem=f"g2_{prefix}_maxp_seed17",
            run_id=f"fullmem_v4_g2_{prefix}_maxp_seed17",
            experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
            method_id="MAX-P_reconstructed",
            num_classes=num_classes,
            new_batch=256,
            replay_batch=10,
            replay=10,
            dataset=dataset,
            v3_dataset=v3,
            notes=(
                f"G2 wide MAX-P_reconstructed on {prefix}: A11 batch=256 replay=10, "
                "Avalanche 0.6.0. Not a paper number. Not aliased from high-batch S*."
            ),
            task_id=f"g2-dev-{prefix.replace('_', '-')}-maxp-seed17",
        )
    )
    return out


def maxa(prefix: str, num_classes: int, dataset, v3: str) -> dict:
    return item(
        stem=f"g2_{prefix}_maxa_seed17",
        run_id=f"fullmem_v4_g2_{prefix}_maxa_seed17",
        experiment_id=f"V4_g2_dev_{prefix}_seed17_wide",
        method_id="MAX-A_reconstructed",
        num_classes=num_classes,
        plugins="gem_ewc",
        extra_algo=MAXA_EXTRA,
        new_batch=1,
        replay_batch=1,
        replay=200,
        dataset=dataset,
        v3_dataset=v3,
        notes=(
            f"G2 wide MAX-A_reconstructed on {prefix}: A11 batch=1 replay=200 GEM+EWC "
            "ewc_lambda=100 patterns_per_exp=50, Avalanche 0.6.0. Not aliased from S*."
        ),
        task_id=f"g2-dev-{prefix.replace('_', '-')}-maxa-seed17",
    )


def clone_orecon_hash(
    src: str,
    stem: str,
    run_id: str,
    experiment_id: str,
    prefetch_on: bool,
    notes: str,
    task_id: str,
    seq: int,
) -> None:
    text = (CFG / src).read_text(encoding="utf-8")
    old_run = None
    for line in text.splitlines():
        if line.startswith("run_id:"):
            old_run = line.split(":", 1)[1].strip()
            break
    if not old_run:
        raise SystemExit(f"no run_id in {src}")
    text = text.replace(f"run_id: {old_run}", f"run_id: {run_id}", 1)
    # claim / experiment
    text = text.replace("experiment_id: V4_g2_dev_cifar100_orecon_full", f"experiment_id: {experiment_id}")
    text = text.replace("experiment_id: V4_g2_dev_core50_nc_orecon_full", f"experiment_id: {experiment_id}")
    text = text.replace("- V4_g2_dev_cifar100_orecon_full", f"- {experiment_id}")
    text = text.replace("- V4_g2_dev_core50_nc_orecon_full", f"- {experiment_id}")
    if prefetch_on:
        text = text.replace(
            "prefetch:\n  enabled: true\n  queue_depth: 2",
            "prefetch:\n  enabled: true\n  queue_depth: 2",
            1,
        )
    else:
        text = text.replace(
            "prefetch:\n  enabled: true\n  queue_depth: 2",
            "prefetch:\n  enabled: false\n  queue_depth: 1",
            1,
        )
        text = text.replace(
            "prefetch:\n  enabled: false\n  queue_depth: 1",
            "prefetch:\n  enabled: false\n  queue_depth: 1",
            1,
        )
    if "data_supply:" not in text:
        text = text.replace(
            "measurement:\n",
            "data_supply:\n  record_hashes: true\n  profile: natural_ondemand\n  profile_path: null\nmeasurement:\n",
            1,
        )
    # notes last quoted line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("notes:"):
            lines[i] = f"notes: {json.dumps(notes)}"
    text = "\n".join(lines) + "\n"
    dest = CFG / f"{stem}.yaml"
    dest.write_text(text, encoding="utf-8", newline="\n")
    print("yaml", dest.name)
    write_task(
        task_id,
        seq,
        "probe",
        ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{stem}.yaml"],
        BATCH,
    )


def specs() -> list[dict]:
    c100 = cifar100_dataset
    c10 = cifar10_dataset
    nc = core50_dataset
    ni = core50_ni_dataset
    nic = core50_nic_dataset

    def endless_ic() -> str:
        return endless_dataset("endless_ic", 4, "Classes")

    def endless_il() -> str:
        return endless_dataset("endless_il", 5, "Illumination")

    def endless_wc() -> str:
        return endless_dataset("endless_wc", 5, "Weather")

    out: list[dict] = []

    # CIFAR100: seed=17 already has S0(hash), LR, plugins-in-flight, S*/Oracle.
    out.append(
        item(
            stem="g2_cifar100_gss_seed17",
            run_id="fullmem_v4_g2_cifar100_gss_seed17",
            experiment_id="V4_g2_dev_cifar100_seed17_wide",
            method_id="GSS",
            num_classes=100,
            algo_base="gss",
            extra_algo=GSS_EXTRA,
            replay_policy="gss_greedy",
            replay_repr="avalanche_gss",
            dataset=c100,
            v3_dataset="splitcifar100",
            notes="G2 wide seed=17 GSS. Seed=0 GSS P=0.01; recheck. Not a freeze.",
            task_id="g2-dev-cifar100-gss-seed17",
        )
    )
    out.append(
        item(
            stem="g2_cifar100_gem_seed17",
            run_id="fullmem_v4_g2_cifar100_gem_seed17",
            experiment_id="V4_g2_dev_cifar100_seed17_wide",
            method_id="GEM",
            num_classes=100,
            algo_base="gem",
            extra_algo=GEM20,
            replay_policy="gem_patterns_per_exp",
            replay_repr="avalanche_gem",
            dataset=c100,
            v3_dataset="splitcifar100",
            notes="G2 wide seed=17 GEM as base on CIFAR100.",
            task_id="g2-dev-cifar100-gem-seed17",
        )
    )
    out.append(
        item(
            stem="g2_cifar100_agem_seed17",
            run_id="fullmem_v4_g2_cifar100_agem_seed17",
            experiment_id="V4_g2_dev_cifar100_seed17_wide",
            method_id="AGEM",
            num_classes=100,
            algo_base="agem",
            extra_algo=AGEM20,
            replay_policy="agem_patterns_per_exp",
            replay_repr="avalanche_agem",
            dataset=c100,
            v3_dataset="splitcifar100",
            notes="G2 wide seed=17 AGEM as base on CIFAR100.",
            task_id="g2-dev-cifar100-agem-seed17",
        )
    )
    out.append(
        item(
            stem="g2_cifar100_maxp_seed17",
            run_id="fullmem_v4_g2_cifar100_maxp_seed17",
            experiment_id="V4_g2_dev_cifar100_seed17_wide",
            method_id="MAX-P_reconstructed",
            num_classes=100,
            new_batch=256,
            replay_batch=10,
            replay=10,
            dataset=c100,
            v3_dataset="splitcifar100",
            notes="G2 wide MAX-P_reconstructed CIFAR100: A11 batch=256 replay=10, Avalanche 0.6.0. Not S*.",
            task_id="g2-dev-cifar100-maxp-seed17",
        )
    )

    # CIFAR10 full method card (no MAX-A; L_cal for O-recon not measured here).
    out.extend(stream("cifar10", 10, c10, "splitcifar10", gem_pat=GEM20, agem_pat=AGEM20))

    # Endless A22 development split: S0 + LR only.
    for name, n_exp, scenario, nclass, v3 in (
        ("endless_ic", 4, "Classes", 5, "endless_ic"),
        ("endless_il", 5, "Illumination", 5, "endless_il"),
        ("endless_wc", 5, "Weather", 5, "endless_wc"),
    ):

        def _ds(name=name, n_exp=n_exp, scenario=scenario) -> str:
            return endless_dataset(name, n_exp, scenario)

        out.append(
            item(
                stem=f"g2_{name}_s0_seed17",
                run_id=f"fullmem_v4_g2_{name}_s0_seed17",
                experiment_id=f"V4_g2_dev_{name}_seed17_wide",
                method_id="S0",
                num_classes=nclass,
                dataset=_ds,
                v3_dataset=v3,
                notes=f"G2 wide seed=17 S0 on {name}. A22 grouped holdout, not official robot protocol.",
                task_id=f"g2-dev-{name.replace('_', '-')}-s0-seed17",
            )
        )
        out.append(
            item(
                stem=f"g2_{name}_lr_seed17",
                run_id=f"fullmem_v4_g2_{name}_lr_seed17",
                experiment_id=f"V4_g2_dev_{name}_seed17_wide",
                method_id="LR_reconstructed",
                num_classes=nclass,
                algo_base="lr",
                dataset=_ds,
                v3_dataset=v3,
                notes=f"G2 wide seed=17 LR reconstructed on {name}. A22 split. Not a freeze.",
                task_id=f"g2-dev-{name.replace('_', '-')}-lr-seed17",
            )
        )

    # NI: S0/LR/plugins (H2 extra stream). Skip GSS/GEM/AGEM-as-base here.
    out.append(
        item(
            stem="g2_core50_ni_s0_seed17",
            run_id="fullmem_v4_g2_core50_ni_s0_seed17",
            experiment_id="V4_g2_dev_core50_ni_seed17_wide",
            method_id="S0",
            num_classes=50,
            dataset=ni,
            v3_dataset="core50_ni",
            notes="G2 wide seed=17 S0 on CORe50-NI.",
            task_id="g2-dev-core50-ni-s0-seed17",
        )
    )
    out.append(
        item(
            stem="g2_core50_ni_lr_seed17",
            run_id="fullmem_v4_g2_core50_ni_lr_seed17",
            experiment_id="V4_g2_dev_core50_ni_seed17_wide",
            method_id="LR_reconstructed",
            num_classes=50,
            algo_base="lr",
            dataset=ni,
            v3_dataset="core50_ni",
            notes="G2 wide seed=17 LR reconstructed on CORe50-NI.",
            task_id="g2-dev-core50-ni-lr-seed17",
        )
    )
    for plug, method, extra in (
        ("gem", "ER+GEM", GEM20),
        ("ewc", "ER+EWC", EWC_EXTRA),
        ("gem_ewc", "ER+GEM+EWC", GEM_EWC20),
    ):
        out.append(
            item(
                stem=f"g2_core50_ni_er_{plug}_seed17",
                run_id=f"fullmem_v4_g2_core50_ni_er_{plug}_seed17",
                experiment_id="V4_g2_dev_core50_ni_seed17_wide",
                method_id=method,
                num_classes=50,
                plugins=plug,
                extra_algo=extra,
                dataset=ni,
                v3_dataset="core50_ni",
                notes=f"G2 wide PLAN C {method} on CORe50-NI, seed=17.",
                task_id=f"g2-dev-core50-ni-er-{plug.replace('_', '-')}-seed17",
            )
        )

    # NC full card except MAX-A (slow tail) and except LR (already done).
    out.append(
        item(
            stem="g2_core50_nc_s0_seed17",
            run_id="fullmem_v4_g2_core50_nc_s0_seed17",
            experiment_id="V4_g2_dev_core50_nc_seed17_wide",
            method_id="S0",
            num_classes=50,
            dataset=nc,
            v3_dataset="core50_nc",
            notes="G2 wide seed=17 S0 on CORe50-NC. Pairing for LR/plugins.",
            task_id="g2-dev-core50-nc-s0-seed17",
        )
    )
    for spec in stream("core50_nc", 50, nc, "core50_nc", gem_pat=GEM50, agem_pat=AGEM50):
        # stream() already includes S0 and LR; skip those duplicates.
        if spec["method_id"] in {"S0", "LR_reconstructed"}:
            continue
        out.append(spec)

    # NIC: S0 + LR only. Plugin/GSS on 79 exp would be many hours.
    out.append(
        item(
            stem="g2_core50_nic_s0_seed17",
            run_id="fullmem_v4_g2_core50_nic_s0_seed17",
            experiment_id="V4_g2_dev_core50_nic_seed17_wide",
            method_id="S0",
            num_classes=50,
            dataset=nic,
            v3_dataset="core50_nic",
            notes="G2 wide seed=17 S0 on CORe50-NIC nicv2_79. Not plugin stack.",
            task_id="g2-dev-core50-nic-s0-seed17",
        )
    )
    out.append(
        item(
            stem="g2_core50_nic_lr_seed17",
            run_id="fullmem_v4_g2_core50_nic_lr_seed17",
            experiment_id="V4_g2_dev_core50_nic_seed17_wide",
            method_id="LR_reconstructed",
            num_classes=50,
            algo_base="lr",
            dataset=nic,
            v3_dataset="core50_nic",
            notes="G2 wide seed=17 LR reconstructed on CORe50-NIC. 79 exp.",
            task_id="g2-dev-core50-nic-lr-seed17",
        )
    )

    # Slow tail: MAX-A reconstructed (batch=1 + GEM/EWC).
    out.append(maxa("cifar100", 100, c100, "splitcifar100"))
    out.append(maxa("cifar10", 10, c10, "splitcifar10"))
    out.append(maxa("core50_nc", 50, nc, "core50_nc"))

    return out


def main() -> None:
    generated = specs()
    seen = set()
    for spec in generated:
        if spec["task_id"] in seen:
            raise SystemExit(f"duplicate task {spec['task_id']}")
        seen.add(spec["task_id"])
        dest = CFG / f"{spec['stem']}.yaml"
        dest.write_text(render(spec), encoding="utf-8", newline="\n")
        print("yaml", dest.name)
        write_task(
            spec["task_id"],
            spec["seq"],
            "probe",
            ["-m", "orion_repro.run", "--config", f"configs/fullmem_v4/{spec['stem']}.yaml"],
            BATCH,
        )

    clone_orecon_hash(
        "g2_cifar100_orecon_full.yaml",
        "g2_cifar100_orecon_full_hash_off_seed17",
        "fullmem_v4_g2_cifar100_orecon_full_hash_off_seed17",
        "V4_g2_dev_cifar100_orecon_h7",
        False,
        "G2 wide H7 O-recon full prefetch off + record_hashes, seed=17. Trajectory may diverge. Not a freeze.",
        "g2-dev-cifar100-orecon-full-hash-off-seed17",
        next_seq(),
    )
    clone_orecon_hash(
        "g2_cifar100_orecon_full.yaml",
        "g2_cifar100_orecon_full_hash_on_seed17",
        "fullmem_v4_g2_cifar100_orecon_full_hash_on_seed17",
        "V4_g2_dev_cifar100_orecon_h7",
        True,
        "G2 wide H7 O-recon full prefetch on + record_hashes, seed=17. Not isolated S0 H7.",
        "g2-dev-cifar100-orecon-full-hash-on-seed17",
        next_seq(),
    )
    clone_orecon_hash(
        "g2_core50_nc_orecon_full.yaml",
        "g2_core50_nc_orecon_full_hash_off_seed17",
        "fullmem_v4_g2_core50_nc_orecon_full_hash_off_seed17",
        "V4_g2_dev_core50_nc_orecon_h7",
        False,
        "G2 wide H7 O-recon full NC prefetch off + record_hashes, seed=17.",
        "g2-dev-core50-nc-orecon-full-hash-off-seed17",
        next_seq(),
    )
    clone_orecon_hash(
        "g2_core50_nc_orecon_full.yaml",
        "g2_core50_nc_orecon_full_hash_on_seed17",
        "fullmem_v4_g2_core50_nc_orecon_full_hash_on_seed17",
        "V4_g2_dev_core50_nc_orecon_h7",
        True,
        "G2 wide H7 O-recon full NC prefetch on + record_hashes, seed=17.",
        "g2-dev-core50-nc-orecon-full-hash-on-seed17",
        next_seq(),
    )
    write_task(
        "g2-occupancy-f",
        next_seq(),
        "probe",
        ["-m", "orion_repro.stages.fullmem_v4", "g2-occupancy"],
        BATCH,
    )
    write_task(
        "g2-cost-c",
        next_seq(),
        "probe",
        ["-m", "orion_repro.stages.fullmem_v4", "g2-cost"],
        BATCH,
    )
    print("tasks", len(seen) + 6)


if __name__ == "__main__":
    main()
