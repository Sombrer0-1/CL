"""Formal matrix emission for effectiveness_v3. Writes only after freeze."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from orion_repro.control.urge import coefficients_from_weights, preference_weights
from orion_repro.runner.spec import validate_mapping
from orion_repro.stages.effectiveness_v3.constants import (
    DEVICE,
    FORMAL_PROTOCOL_ID,
    GROUP_METHODS,
    GROUP_VARIANTS,
    PLUGIN_DEFAULTS,
    PRIMARY_DATASETS,
    SEEDS,
    SLOT_COUNTS,
    STUDY_ID,
    TOTAL_SLOTS,
)
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json, load_yaml, write_yaml

FORMAL_TEMPLATES = {
    "core50_nc": "configs/formal/e03/core50_nc/orion_formula/seed0.yaml",
    "splitcifar100": "configs/formal/e03/splitcifar100/orion_formula/seed0.yaml",
}


def interleave_methods(methods: tuple[str, ...] | list[str], seeds: tuple[int, ...] = SEEDS) -> list[dict[str, Any]]:
    methods = list(methods)
    order = []
    for i, seed in enumerate(seeds):
        rotated = methods[i % len(methods) :] + methods[: i % len(methods)]
        for method in rotated:
            order.append({"seed": int(seed), "method": method})
    return order


def cell_id(group: str, dataset: str, scenario: str, method: str, seed: int, variant: str = "default") -> str:
    return f"{group}/{dataset}/{scenario}/{method}/{seed}/{variant}"


def _base_formal(root: Path, dataset: str, seed: int, frozen: dict[str, Any]) -> dict[str, Any]:
    spec = load_yaml(root / FORMAL_TEMPLATES[dataset])
    spec.update(
        study_id=STUDY_ID,
        reuse_version=4,
        protocol_id=FORMAL_PROTOCOL_ID,
        phase="formal",
        alignment_version="effectiveness_v3",
        frozen_hash=frozen["frozen_hash"],
        frozen_protocol_path=f"experiments/{STUDY_ID}/revisions/{frozen['revision']}/frozen_protocol.json",
        dataset_manifest_sha256=frozen["data_hashes"][
            "core50_nc_formal" if dataset == "core50_nc" else "splitcifar100_formal"
        ],
        code_revision_or_snapshot=frozen["source_hash"],
        environment_lock_sha256=frozen["environment_lock_sha256"],
    )
    spec["seeds"] = dict.fromkeys(("model", "stream", "replay", "augmentation"), int(seed))
    spec["training"]["device"] = DEVICE
    spec["algorithm"].update(PLUGIN_DEFAULTS)
    spec["controller"]["m_batch"] = frozen["m_batch"]
    spec["controller"]["m_frame"] = frozen["m_frame"]
    spec["budget"].update(
        enforcement="device_allocator_enforced",
        controlled_resource="device",
        guard_mode="stop",
    )
    spec["prefetch"].update(enabled=False, queue_depth=1, pin_memory=False, num_workers=0)
    spec["resource_envelope"] = {"enabled": False, "reserved_bytes_by_experience": [0] * int(spec["dataset"]["n_experiences"])}
    spec["data_supply"] = {"record_hashes": False, "profile": "default"}
    spec["controller"]["feedback_source"] = "official_test_seen"
    spec["controller"]["thr0"] = frozen["thr0"]
    spec["controller"]["delta"] = frozen["delta"]
    spec["controller"]["updates"] = {"alpha": frozen["alpha"], "beta": frozen["beta"]}
    spec["controller"]["mb0"] = float(frozen["initial_counts"]["new_batch"])
    spec["controller"]["mr0"] = float(frozen["initial_counts"]["replay_capacity"])
    spec["training"]["new_batch"] = frozen["initial_counts"]["new_batch"]
    spec["training"]["replay_batch"] = frozen["initial_counts"]["new_batch"]
    spec["replay"]["capacity"] = frozen["initial_counts"]["replay_capacity"]
    spec["training"]["eval_batch"] = int(frozen["datasets"][dataset]["eval_batch"])
    return spec


def _apply_orion(spec: dict[str, Any], *, plugins: bool, policy: str, enabled: bool, start_enabled: bool | None = None) -> None:
    spec["controller"]["enabled"] = bool(enabled)
    spec["controller"]["plugin_policy"] = policy
    if plugins:
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"]["optional_start_enabled"] = bool(start_enabled if start_enabled is not None else policy == "fixed_advanced")
        spec["algorithm"].update(PLUGIN_DEFAULTS)
    else:
        spec["algorithm"]["optional_plugins"] = "none"
        spec["algorithm"]["optional_start_enabled"] = False


def _set_quota(spec: dict[str, Any], quota_bytes: int, m_max_mib: float) -> None:
    spec["budget"]["limit_bytes"] = int(quota_bytes)
    spec["controller"]["thresholds"]["m_max_mib"] = float(m_max_mib)


def _set_latency(spec: dict[str, Any], seconds: float) -> None:
    spec["controller"]["thresholds"]["latency_s"] = float(seconds)


def _cell_spec(
    frozen: dict[str, Any],
    *,
    group: str,
    dataset: str,
    method: str,
    seed: int,
    variant: str,
    root: Path,
) -> dict[str, Any]:
    spec = _base_formal(root, dataset, seed, frozen)
    ds = frozen["datasets"][dataset]
    spec["v3_group"] = group
    spec["v3_variant"] = variant
    spec["method_id"] = method
    spec["experiment_id"] = f"V3_{group}_{dataset}_{method}"
    spec["claim_ids"] = [f"V3_{group}"]
    if group in {"A", "D", "G"}:
        scenario = "tight"
        _set_quota(spec, ds["quota_tight_bytes"], ds["quota_tight_mib"])
        _set_latency(spec, ds["latency_cal_s"])
    elif group in {"B", "F"}:
        scenario = "loose"
        _set_quota(spec, ds["quota_loose_bytes"], ds["quota_loose_mib"])
        _set_latency(spec, ds["latency_cal_s"])
    elif group == "C":
        scenario = "dynamic"
        _set_quota(spec, frozen["dyn_quota_bytes"], frozen["dyn_quota_bytes"] / (1024**2))
        spec["resource_envelope"] = {
            "enabled": True,
            "schedule_path": f"experiments/{STUDY_ID}/revisions/{frozen['revision']}/dyn_schedule.json",
            "reserved_bytes_by_experience": frozen["reserved_bytes_by_experience"],
        }
        _set_latency(spec, ds["latency_cal_s"])
    elif group == "E":
        scenario = "io"
        _set_quota(spec, ds["quota_loose_bytes"], ds["quota_loose_mib"])
        _set_latency(spec, ds["latency_cal_s"])
    elif group == "H":
        scenario = "host"
        _set_quota(spec, ds["quota_tight_bytes"], ds["quota_tight_mib"])
        _set_latency(spec, ds["latency_cal_s"])
    else:
        raise StageError(group)
    spec["v3_scenario"] = scenario
    spec["v3_cell_id"] = cell_id(group, dataset, scenario, method, seed, variant)

    if method == "S0":
        _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
    elif method in {"S_star", "S_star_dynamic", "S_star_host"}:
        _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
        key = "dyn" if method == "S_star_dynamic" else ("tight" if scenario in {"tight", "host", "dynamic"} and method != "S_star" else scenario if scenario in {"tight", "loose"} else "tight")
        if method == "S_star_dynamic":
            key = "dyn"
        elif method == "S_star_host":
            key = "tight"
        elif scenario == "loose":
            key = "loose"
        else:
            key = "tight"
        chosen = ds["static_selections"][key]
        spec["training"]["new_batch"] = chosen["new_batch"]
        spec["training"]["replay_batch"] = chosen["new_batch"]
        spec["replay"]["capacity"] = chosen["replay_capacity"]
        spec["static_selection_id"] = chosen["config_id"]
    elif method == "O00":
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        _set_latency(spec, frozen["o00_latency_s"])
        spec["controller"]["thresholds"]["m_max_mib"] = frozen["o00_memory_mib"]
    elif method == "O10":
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        spec["controller"]["thresholds"]["m_max_mib"] = frozen["o00_memory_mib"]
    elif method == "O01":
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        _set_latency(spec, frozen["o00_latency_s"])
    elif method in {"O11", "orion_off", "orion_on", "orion_host"}:
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
    elif method == "R11":
        _apply_orion(spec, plugins=True, policy="fixed_default", enabled=True, start_enabled=False)
    elif method == "F11":
        _apply_orion(spec, plugins=True, policy="fixed_advanced", enabled=True, start_enabled=True)
    elif method == "O11_half_life":
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        spec["controller"]["delta"] = float(frozen["decay_delta"])
    elif method == "agem_static":
        _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
        spec["algorithm"]["base"] = "agem"
        spec["algorithm"]["optional_plugins"] = "none"
        spec["algorithm"].update(PLUGIN_DEFAULTS)
    elif method == "agem_adaptive_ewc":
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        spec["algorithm"]["base"] = "agem"
        spec["algorithm"]["optional_plugins"] = "ewc"
        spec["algorithm"]["optional_start_enabled"] = False
        spec["algorithm"].update(PLUGIN_DEFAULTS)
    elif method in {"static_off", "static_on"}:
        _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
    else:
        raise StageError(method)

    if group == "D" and variant in frozen["preference_orders"] and frozen["preference_orders"][variant]:
        order = list(frozen["preference_orders"][variant])
        spec["controller"]["preference"] = "ranked"
        spec["controller"]["preference_order"] = order
        spec["controller"]["coefficients"] = coefficients_from_weights(preference_weights(order))
    if group == "G":
        _apply_sensitivity(spec, variant, frozen, ds)
    if group == "E":
        on = method.endswith("_on")
        spec["data_supply"] = {
            "record_hashes": True,
            "profile": "natural_ondemand",
            "profile_path": f"experiments/{STUDY_ID}/revisions/{frozen['revision']}/io_profile.json",
        }
        spec["prefetch"].update(
            enabled=on,
            queue_depth=int(frozen["io_prefetch"]["queue_depth"]),
            pin_memory=bool(frozen["io_prefetch"]["pin_memory"]),
            num_workers=int(frozen["io_prefetch"]["num_workers"]),
        )
        spec["training"]["deterministic_algorithms"] = method.startswith("static")
    validate_mapping(spec, require_provenance=False)
    return spec


def _apply_sensitivity(spec: dict[str, Any], variant: str, frozen: dict[str, Any], ds: dict[str, Any]) -> None:
    if variant == "thr_half":
        spec["controller"]["thr0"] = frozen["thr0"] * 0.5
    elif variant == "thr_double":
        spec["controller"]["thr0"] = frozen["thr0"] * 2.0
    elif variant == "alpha_double":
        spec["controller"]["updates"]["alpha"] = frozen["alpha"] * 2.0
    elif variant == "beta_double":
        spec["controller"]["updates"]["beta"] = frozen["beta"] * 2.0
    elif variant == "lr_half":
        spec["training"]["optimizer"]["lr"] = float(spec["training"]["optimizer"]["lr"]) * 0.5
    elif variant == "initial_batch32":
        spec["training"]["new_batch"] = 32
        spec["training"]["replay_batch"] = 32
        spec["controller"]["mb0"] = 32.0
    elif variant == "initial_replay1000":
        spec["replay"]["capacity"] = 1000
        spec["controller"]["mr0"] = 1000.0
    elif variant == "quota_mid":
        _set_quota(spec, int(ds["quota_mid_mib"]) * 1024**2, ds["quota_mid_mib"])
    else:
        raise StageError(variant)


def planned_cells(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    cells = []
    coverage = frozen.get("scenario_coverage") or {}
    io_ok = (frozen.get("io_prefetch") or {}).get("constructed")
    host_ok = (frozen.get("host") or {}).get("status") in {"available", "realized"}
    dyn_ok = coverage.get("dynamic_reservation") == "realized"
    for group, methods in GROUP_METHODS.items():
        datasets = list(PRIMARY_DATASETS) if group in {"A", "B"} else ["core50_nc"]
        variants = GROUP_VARIANTS.get(group) or ("default",)
        scenario = {
            "A": "tight",
            "B": "loose",
            "C": "dynamic",
            "D": "tight",
            "E": "io",
            "F": "loose",
            "G": "tight",
            "H": "host",
        }[group]
        for dataset in datasets:
            if group in {"D", "G"}:
                for variant in variants:
                    for item in interleave_methods(methods):
                        cells.append(
                            {
                                "group": group,
                                "dataset": dataset,
                                "scenario": scenario,
                                "method": item["method"],
                                "seed": item["seed"],
                                "variant": variant,
                            }
                        )
            else:
                for item in interleave_methods(methods):
                    cells.append(
                        {
                            "group": group,
                            "dataset": dataset,
                            "scenario": scenario,
                            "method": item["method"],
                            "seed": item["seed"],
                            "variant": "default",
                        }
                    )
    if len(cells) != TOTAL_SLOTS:
        raise StageError(f"planned cells {len(cells)} != {TOTAL_SLOTS}")
    eligible = []
    excluded = []
    for cell in cells:
        reason = None
        if cell["group"] == "E" and not io_ok:
            reason = "scenario_not_realized:io"
        if cell["group"] == "H" and not host_ok:
            reason = "scenario_not_realized:host"
        if cell["group"] == "C" and not dyn_ok:
            reason = "scenario_not_realized:dynamic_reservation"
        cell = dict(cell)
        cell["cell_id"] = cell_id(
            cell["group"], cell["dataset"], cell["scenario"], cell["method"], cell["seed"], cell["variant"]
        )
        if reason:
            cell["exclude_reason"] = reason
            excluded.append(cell)
        else:
            eligible.append(cell)
    return eligible, excluded, cells


def emit(frozen: dict[str, Any], context: StageContext) -> dict[str, Any]:
    context.reject_foreign_study(frozen)
    if frozen.get("frozen_hash") is None:
        raise StageError("refuse to emit without frozen_hash")
    eligible, excluded, all_cells = planned_cells(frozen)
    rels = []
    definitions = []
    for cell in eligible:
        spec = _cell_spec(
            frozen,
            group=cell["group"],
            dataset=cell["dataset"],
            method=cell["method"],
            seed=cell["seed"],
            variant=cell["variant"],
            root=context.root,
        )
        rel = (
            f"configs/{STUDY_ID}/{context.revision}/formal/"
            f"{cell['group']}_{cell['dataset']}_{cell['method']}_s{cell['seed']}_{cell['variant']}.yaml"
        )
        path = context.root / rel
        context.assert_output_path(path)
        write_yaml(path, spec)
        rels.append(rel)
        item = dict(cell)
        item["config"] = rel
        definitions.append(item)
    for cell in excluded:
        definitions.append(dict(cell))
    counts = {name: 0 for name in SLOT_COUNTS}
    for cell in all_cells:
        counts[cell["group"]] += 1
    if counts != SLOT_COUNTS:
        raise StageError(f"group counts {counts} != {SLOT_COUNTS}")
    matrix = {
        "kind": "matrix_manifest",
        "study_id": STUDY_ID,
        "revision": context.revision,
        "frozen_hash": frozen["frozen_hash"],
        "executable": True,
        "design_slots": TOTAL_SLOTS,
        "eligible_n": len(eligible),
        "excluded_n": len(excluded),
        "group_slots": SLOT_COUNTS,
        "cells": definitions,
        "order": [c["cell_id"] for c in eligible],
        "eligible_cells": [c["cell_id"] for c in eligible],
        "excluded_cells": excluded,
        "configs": rels,
    }
    tmp_dir = context.revision_dir / "_matrix_tmp"
    context.assert_output_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / "matrix.json"
    atomic_write_json(tmp_path, matrix)
    final = context.revision_dir / "matrix.json"
    context.assert_output_path(final)
    if final.exists():
        import json

        old = json.loads(final.read_text(encoding="utf-8"))
        if old.get("frozen_hash") != matrix["frozen_hash"]:
            raise StageError("existing matrix frozen_hash mismatch")
    else:
        tmp_path.replace(final)
    schedule = {
        "reserved_bytes_by_experience": frozen["reserved_bytes_by_experience"],
        "note": "same-process CUDA uint8 reservation; not model or replay bytes",
    }
    sched_path = context.revision_dir / "dyn_schedule.json"
    context.assert_output_path(sched_path)
    atomic_write_json(sched_path, schedule)
    io_path = context.revision_dir / "io_profile.json"
    context.assert_output_path(io_path)
    atomic_write_json(io_path, dict(frozen["io_prefetch"]))
    return matrix
