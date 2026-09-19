"""Development probe planning, spec emission, and evidence collection."""

from __future__ import annotations

import copy
import csv
import json
import math
from statistics import median
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orion_repro.runner.spec import canonical_hash, validate_mapping
from orion_repro.stages.effectiveness_v3.constants import (
    DEVELOPMENT_PROTOCOL_ID,
    DEVICE,
    EVAL_BATCH_CANDIDATES,
    PLUGIN_DEFAULTS,
    PRIMARY_DATASETS,
    STATIC_GRID,
    STUDY_ID,
)
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.util import StageError, load_yaml, sha256_file, write_yaml

N_EXPECTED = {"core50_nc": 9, "splitcifar100": 10}
TEMPLATE = {
    "core50_nc": "configs/development/core50_nc_er_static.yaml",
    "splitcifar100": "configs/development/cifar100_er_static.yaml",
}


@dataclass
class ProbeRequest:
    probe_id: str
    role: str
    dataset: str
    full_stream: bool
    n_expected: int
    spec: dict[str, Any]
    rel_config: str


@dataclass
class ProbeBatch:
    probes: list[ProbeRequest] = field(default_factory=list)
    blocked_reason: str | None = None


def _template(root: Path, dataset: str) -> dict[str, Any]:
    rel = TEMPLATE.get(dataset)
    if not rel:
        raise StageError(f"unknown dataset {dataset}")
    return load_yaml(root / rel)


def stamp_dev(spec: dict[str, Any], *, role: str, method: str, dataset: str) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec.update(
        study_id=STUDY_ID,
        reuse_version=4,
        protocol_id=DEVELOPMENT_PROTOCOL_ID,
        phase="development",
        alignment_version="effectiveness_v3",
        experiment_id=f"V3_{role}",
        claim_ids=[f"V3_{role}"],
        method_id=method,
        pressure_role=role,
        v3_role=role,
        v3_dataset=dataset,
    )
    spec["training"]["device"] = DEVICE
    spec["controller"]["feedback_source"] = "development_val_seen"
    spec["controller"]["plugin_policy"] = spec["controller"].get("plugin_policy", "adaptive")
    spec["prefetch"].update(enabled=False, queue_depth=1, pin_memory=False, num_workers=0)
    spec["measurement"]["checkpoint_policy"] = "none"
    return spec


def apply_quota(spec: dict[str, Any], quota_mib: int | None, *, enforcement: str | None = None) -> None:
    if quota_mib is None:
        spec["budget"].update(
            enforcement="observed_only",
            controlled_resource="device",
            limit_bytes=None,
        )
        return
    spec["budget"].update(
        enforcement=enforcement or "device_allocator_enforced",
        controlled_resource="device",
        limit_bytes=int(quota_mib) * 1024**2,
        guard_mode="stop",
    )
    spec["controller"]["thresholds"]["m_max_mib"] = float(quota_mib)


def completed_roles(manifest: dict[str, Any]) -> set[tuple[str, str]]:
    out = set()
    for row in manifest.get("runs") or []:
        if row.get("status") in {"completed", "cuda_oom", "host_oom", "budget_exceeded"}:
            out.add((str(row.get("role")), str(row.get("dataset"))))
    return out


def _role_complete(manifest: dict[str, Any], role: str, dataset: str, n: int) -> bool:
    rows = [
        r
        for r in manifest.get("runs") or []
        if r.get("role") == role and r.get("dataset") == dataset and r.get("status") not in {None, "not_run", "interrupted"}
    ]
    return len(rows) >= n


def _l_cal_s(evidence: dict[str, Any], dataset: str) -> float:
    learning: list[float] = []
    for row in evidence.get("runs") or []:
        if row.get("role") != "l_cal" or row.get("dataset") != dataset:
            continue
        if row.get("status") != "completed":
            continue
        learning.extend(float(v) for v in (row.get("learning_s_by_experience") or []))
    return float(median(learning)) if learning else 30.0


def plan_probes(
    design: dict[str, Any],
    evidence: dict[str, Any],
    context: StageContext,
    *,
    identity_recalibrate: bool = False,
) -> ProbeBatch:
    context.reject_foreign_study(design)
    if evidence:
        context.reject_foreign_study(evidence)
    batch = ProbeBatch()
    root = context.root
    for dataset in PRIMARY_DATASETS:
        n_exp = N_EXPECTED[dataset]
        if not _role_complete(evidence, "eval_batch", dataset, len(EVAL_BATCH_CANDIDATES)):
            for eval_batch in EVAL_BATCH_CANDIDATES:
                spec = stamp_dev(_template(root, dataset), role="eval_batch", method=f"eval{eval_batch}", dataset=dataset)
                spec["training"]["eval_batch"] = int(eval_batch)
                spec["dataset"]["experience_limit"] = 1
                apply_quota(spec, 1024)
                probe_id = f"eval_{dataset}_b{eval_batch}"
                batch.probes.append(_request(context, spec, probe_id, "eval_batch", dataset, False, 1))
            return batch
    eval_choice = chosen_eval_batch(evidence)
    if any(v is None for v in eval_choice.values()):
        batch.blocked_reason = "eval_batch not yet feasible on both datasets"
        return batch
    for dataset in PRIMARY_DATASETS:
        if not _role_complete(evidence, "s0_profile", dataset, 2):
            for rep in range(2):
                spec = stamp_dev(_template(root, dataset), role="s0_profile", method=f"s0_rep{rep}", dataset=dataset)
                spec["training"]["eval_batch"] = int(eval_choice[dataset])
                spec["dataset"]["experience_limit"] = N_EXPECTED[dataset]
                spec["seeds"] = {"model": int(rep), "stream": 0, "replay": 0, "augmentation": 0}
                apply_quota(spec, 8192)
                probe_id = f"s0_{dataset}_rep{rep}"
                batch.probes.append(
                    _request(context, spec, probe_id, "s0_profile", dataset, True, N_EXPECTED[dataset])
                )
            return batch
        if identity_recalibrate:
            continue
        if not _role_complete(evidence, "plugin_cost", dataset, 5):
            for name, extra in (
                ("b64", {"training": {"new_batch": 64, "replay_batch": 64}}),
                ("b256", {"training": {"new_batch": 256, "replay_batch": 256}}),
                ("gem", {"algorithm": {"optional_plugins": "gem", "optional_start_enabled": True, **PLUGIN_DEFAULTS}}),
                ("ewc", {"algorithm": {"optional_plugins": "ewc", "optional_start_enabled": True, **PLUGIN_DEFAULTS}}),
                ("gem_ewc", {"algorithm": {"optional_plugins": "gem_ewc", "optional_start_enabled": True, **PLUGIN_DEFAULTS}}),
            ):
                spec = stamp_dev(_template(root, dataset), role="plugin_cost", method=name, dataset=dataset)
                spec["training"]["eval_batch"] = int(eval_choice[dataset])
                apply_quota(spec, 8192)
                for key, value in extra.items():
                    if isinstance(value, dict) and isinstance(spec.get(key), dict):
                        spec[key].update(value)
                    else:
                        spec[key] = value
                batch.probes.append(
                    _request(context, spec, f"cost_{dataset}_{name}", "plugin_cost", dataset, True, N_EXPECTED[dataset])
                )
            return batch
    for dataset in PRIMARY_DATASETS:
        if _chosen_tight(evidence, dataset) is not None:
            continue
        next_quotas, blocked = _next_quota_scan(evidence, dataset)
        if blocked:
            batch.blocked_reason = blocked
            return batch
        if not next_quotas:
            batch.blocked_reason = f"Q_tight not identifiable for {dataset}"
            return batch
        for quota in next_quotas:
            spec = stamp_dev(_template(root, dataset), role="quota_scan", method=f"q{quota}", dataset=dataset)
            spec["training"]["eval_batch"] = int(eval_choice[dataset])
            apply_quota(spec, quota)
            batch.probes.append(
                _request(context, spec, f"quota_{dataset}_q{quota}", "quota_scan", dataset, True, N_EXPECTED[dataset])
            )
        return batch
    for dataset in PRIMARY_DATASETS:
        q_tight = _chosen_tight(evidence, dataset)
        if q_tight is None:
            batch.blocked_reason = f"Q_tight not identifiable for {dataset}"
            return batch
        q_loose = _verified_loose(evidence, dataset, q_tight)
        if q_loose is None:
            next_loose, blocked = _next_loose_quota(evidence, dataset, q_tight)
            if blocked:
                batch.blocked_reason = blocked
                return batch
            spec = stamp_dev(_template(root, dataset), role="q_loose_verify", method=f"loose{next_loose}", dataset=dataset)
            spec["training"]["eval_batch"] = int(eval_choice[dataset])
            apply_quota(spec, next_loose)
            batch.probes.append(
                _request(context, spec, f"loose_{dataset}_q{next_loose}", "q_loose_verify", dataset, True, N_EXPECTED[dataset])
            )
            return batch
        if (not identity_recalibrate) and (not _role_complete(evidence, "plugin_loose", dataset, 1)):
            spec = stamp_dev(_template(root, dataset), role="plugin_loose", method="gem_ewc_on", dataset=dataset)
            spec["training"]["eval_batch"] = int(eval_choice[dataset])
            spec["algorithm"]["optional_plugins"] = "gem_ewc"
            spec["algorithm"]["optional_start_enabled"] = True
            spec["algorithm"].update(PLUGIN_DEFAULTS)
            spec["controller"].update(enabled=False, plugin_policy="fixed_advanced")
            apply_quota(spec, q_loose)
            batch.probes.append(
                _request(
                    context,
                    spec,
                    f"plugin_loose_{dataset}_q{q_loose}",
                    "plugin_loose",
                    dataset,
                    True,
                    N_EXPECTED[dataset],
                )
            )
            return batch
        if not _role_complete(evidence, "l_cal", dataset, 2):
            for rep in range(2):
                spec = stamp_dev(_template(root, dataset), role="l_cal", method=f"lcal{rep}", dataset=dataset)
                spec["training"]["eval_batch"] = int(eval_choice[dataset])
                apply_quota(spec, q_loose)
                spec["seeds"] = {"model": int(rep), "stream": 0, "replay": 0, "augmentation": 0}
                batch.probes.append(
                    _request(context, spec, f"lcal_{dataset}_q{q_loose}_rep{rep}", "l_cal", dataset, True, N_EXPECTED[dataset])
                )
            return batch
        if not _role_complete(evidence, "static_search_tight", dataset, 6):
            for batch_n, replay in STATIC_GRID:
                spec = stamp_dev(
                    _template(root, dataset),
                    role="static_search_tight",
                    method=f"b{batch_n}_r{replay}",
                    dataset=dataset,
                )
                spec["training"].update(new_batch=batch_n, replay_batch=batch_n, eval_batch=int(eval_choice[dataset]))
                spec["replay"]["capacity"] = replay
                apply_quota(spec, q_tight)
                batch.probes.append(
                    _request(
                        context,
                        spec,
                        f"search_tight_{dataset}_b{batch_n}_r{replay}_q{q_tight}",
                        "static_search_tight",
                        dataset,
                        True,
                        N_EXPECTED[dataset],
                    )
                )
            return batch
        if not _role_complete(evidence, "static_search_loose", dataset, 6):
            for batch_n, replay in STATIC_GRID:
                spec = stamp_dev(
                    _template(root, dataset),
                    role="static_search_loose",
                    method=f"b{batch_n}_r{replay}",
                    dataset=dataset,
                )
                spec["training"].update(new_batch=batch_n, replay_batch=batch_n, eval_batch=int(eval_choice[dataset]))
                spec["replay"]["capacity"] = replay
                apply_quota(spec, q_loose)
                batch.probes.append(
                    _request(
                        context,
                        spec,
                        f"search_loose_{dataset}_b{batch_n}_r{replay}_q{q_loose}",
                        "static_search_loose",
                        dataset,
                        True,
                        N_EXPECTED[dataset],
                    )
                )
            return batch
    nc_eval = int(eval_choice["core50_nc"])
    nc_tight = _chosen_tight(evidence, "core50_nc")
    if nc_tight is None:
        batch.blocked_reason = "Q_tight not identifiable for core50_nc"
        return batch
    nc_loose = _verified_loose(evidence, "core50_nc", nc_tight) or (2 * int(nc_tight))
    if identity_recalibrate:
        return _plan_identity_coverage_stamps(
            evidence, context, eval_choice, int(nc_eval), int(nc_tight), int(nc_loose)
        )
    for dataset in PRIMARY_DATASETS:
        if _role_complete(evidence, "control_2x2", dataset, 4):
            continue
        q_ds = _chosen_tight(evidence, dataset)
        if q_ds is None:
            batch.blocked_reason = f"Q_tight not identifiable for {dataset}"
            return batch
        l_cal = _l_cal_s(evidence, dataset)
        eval_b = int(eval_choice[dataset])
        n_exp = N_EXPECTED[dataset]
        for name, latency, m_max in (
            ("O00", 30.0, 4096.0),
            ("O10", float(l_cal), 4096.0),
            ("O01", 30.0, float(q_ds)),
            ("O11", float(l_cal), float(q_ds)),
        ):
            spec = stamp_dev(_template(root, dataset), role="control_2x2", method=name, dataset=dataset)
            spec["training"]["eval_batch"] = eval_b
            spec["controller"].update(enabled=True, plugin_policy="adaptive")
            spec["algorithm"]["optional_plugins"] = "gem_ewc"
            spec["algorithm"]["optional_start_enabled"] = False
            spec["algorithm"].update(PLUGIN_DEFAULTS)
            apply_quota(spec, q_ds)
            spec["controller"]["thresholds"].update(p=0.5, s=0.5, latency_s=latency, m_max_mib=m_max)
            batch.probes.append(
                _request(context, spec, f"control_{dataset}_{name}_q{q_ds}", "control_2x2", dataset, True, n_exp)
            )
        return batch
    if not _role_complete(evidence, "dyn_reservation_probe", "core50_nc", 1):
        peak = _profile_reserved_mib(evidence, "core50_nc") or 0.0
        margin = max(4, int(nc_tight - peak))
        r_mib = max(4, (margin // 8) * 4) if margin >= 8 else 4
        reserved = [0, 0, 0, r_mib * 1024**2, r_mib * 1024**2, r_mib * 1024**2, 0, 0, 0]
        spec = stamp_dev(_template(root, "core50_nc"), role="dyn_reservation_probe", method=f"R{r_mib}", dataset="core50_nc")
        spec["training"]["eval_batch"] = nc_eval
        apply_quota(spec, nc_tight)
        spec["resource_envelope"] = {
            "enabled": True,
            "reserved_bytes_by_experience": reserved,
        }
        spec["controller"].update(enabled=True, plugin_policy="adaptive")
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"].update(PLUGIN_DEFAULTS)
        spec["controller"]["thresholds"].update(
            latency_s=_l_cal_s(evidence, "core50_nc"),
            m_max_mib=float(nc_tight),
        )
        batch.probes.append(_request(context, spec, f"dyn_core50_nc_R{r_mib}", "dyn_reservation_probe", "core50_nc", True, 9))
        return batch
    if _need_dyn_min_positive(evidence):
        reserved = [0, 0, 0, 4 * 1024**2, 4 * 1024**2, 4 * 1024**2, 0, 0, 0]
        spec = stamp_dev(_template(root, "core50_nc"), role="dyn_min_positive", method="R4", dataset="core50_nc")
        spec["training"]["eval_batch"] = nc_eval
        apply_quota(spec, nc_tight)
        spec["resource_envelope"] = {"enabled": True, "reserved_bytes_by_experience": reserved}
        spec["controller"].update(enabled=True, plugin_policy="adaptive")
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"].update(PLUGIN_DEFAULTS)
        spec["controller"]["thresholds"].update(
            latency_s=_l_cal_s(evidence, "core50_nc"),
            m_max_mib=float(nc_tight),
        )
        batch.probes.append(_request(context, spec, "dyn_core50_nc_R4_min_positive", "dyn_min_positive", "core50_nc", True, 9))
        return batch
    dyn = _dyn_anchor(evidence)
    if dyn is not None and not _role_complete(evidence, "static_search_dyn", "core50_nc", 6):
        for batch_n, replay in STATIC_GRID:
            spec = stamp_dev(_template(root, "core50_nc"), role="static_search_dyn", method=f"b{batch_n}_r{replay}", dataset="core50_nc")
            spec["training"].update(new_batch=batch_n, replay_batch=batch_n, eval_batch=nc_eval)
            spec["replay"]["capacity"] = replay
            apply_quota(spec, int(dyn["quota_mib"]))
            spec["resource_envelope"] = {"enabled": True, "reserved_bytes_by_experience": list(dyn["reserved_bytes_by_experience"])}
            batch.probes.append(
                _request(
                    context,
                    spec,
                    f"search_dyn_core50_nc_b{batch_n}_r{replay}_q{nc_tight}",
                    "static_search_dyn",
                    "core50_nc",
                    True,
                    9,
                )
            )
        return batch
    if not (
        _role_complete(evidence, "io_off", "core50_nc", 1)
        and _role_complete(evidence, "io_on", "core50_nc", 1)
    ):
        for enabled, role in ((False, "io_off"), (True, "io_on")):
            spec = stamp_dev(_template(root, "core50_nc"), role=role, method=role, dataset="core50_nc")
            spec["training"]["eval_batch"] = nc_eval
            apply_quota(spec, nc_loose)
            spec["prefetch"].update(enabled=enabled, queue_depth=2, pin_memory=False, num_workers=0)
            spec["data_supply"] = {"record_hashes": True, "profile": "natural_ondemand"}
            batch.probes.append(_request(context, spec, f"{role}_core50_nc", role, "core50_nc", True, 9))
        return batch
    if not _role_complete(evidence, "sensitivity", "core50_nc", len(planned_sensitivity_variants(evidence, "core50_nc"))):
        l_cal = _l_cal_s(evidence, "core50_nc")
        variants = planned_sensitivity_variants(evidence, "core50_nc")
        for name, extra in variants.items():
            spec = stamp_dev(_template(root, "core50_nc"), role="sensitivity", method=name, dataset="core50_nc")
            spec["training"]["eval_batch"] = nc_eval
            spec["algorithm"]["optional_plugins"] = "gem_ewc"
            spec["algorithm"].update(PLUGIN_DEFAULTS)
            spec["controller"].update(enabled=True, plugin_policy="adaptive")
            spec["controller"]["thresholds"].update(latency_s=float(l_cal), m_max_mib=float(nc_tight))
            quota = nc_tight
            extra = dict(extra)
            quota_kind = extra.pop("_quota", None)
            quota_mib = extra.pop("_quota_mib", None)
            if quota_kind == "loose":
                quota = int(nc_loose)
            elif quota_kind == "mid":
                quota = int(quota_mib)
            apply_quota(spec, quota)
            for key, value in extra.items():
                if isinstance(value, dict) and isinstance(spec.get(key), dict):
                    spec[key].update(value)
                else:
                    spec[key] = value
            if quota_kind:
                spec["controller"]["thresholds"]["m_max_mib"] = float(quota)
            batch.probes.append(_request(context, spec, f"sens_{name}_q{quota}", "sensitivity", "core50_nc", True, 9))
        return batch
    if not _role_complete(evidence, "agem_profile", "core50_nc", 2):
        for name, extra in (
            ("agem_static", {"algorithm": {"base": "agem", "optional_plugins": "none", **PLUGIN_DEFAULTS}, "controller": {"enabled": False}}),
            ("agem_adaptive_ewc", {"algorithm": {"base": "agem", "optional_plugins": "ewc", "optional_start_enabled": False, **PLUGIN_DEFAULTS}, "controller": {"enabled": True, "plugin_policy": "adaptive"}}),
        ):
            spec = stamp_dev(_template(root, "core50_nc"), role="agem_profile", method=name, dataset="core50_nc")
            spec["training"]["eval_batch"] = nc_eval
            apply_quota(spec, nc_loose)
            for key, value in extra.items():
                if isinstance(value, dict) and isinstance(spec.get(key), dict):
                    spec[key].update(value)
                else:
                    spec[key] = value
            batch.probes.append(_request(context, spec, f"{name}_q{nc_loose}", "agem_profile", "core50_nc", True, 9))
        return batch
    return batch


def _plan_identity_coverage_stamps(
    evidence: dict[str, Any],
    context: StageContext,
    eval_choice: dict[str, int],
    nc_eval: int,
    nc_tight: int,
    nc_loose: int,
) -> ProbeBatch:
    """S01/S04 identity stamps and S07 IO pair after Q/L/S* probes. Does not rebuild DYN at mid/loose."""
    batch = ProbeBatch()
    root = context.root
    l_cal = _l_cal_s(evidence, "core50_nc")
    if not _role_complete(evidence, "identity_s01_stamp", "core50_nc", 1):
        spec = stamp_dev(_template(root, "core50_nc"), role="identity_s01_stamp", method="O11", dataset="core50_nc")
        spec["training"]["eval_batch"] = nc_eval
        spec["controller"].update(enabled=True, plugin_policy="adaptive")
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"]["optional_start_enabled"] = False
        spec["algorithm"].update(PLUGIN_DEFAULTS)
        apply_quota(spec, nc_tight)
        spec["controller"]["thresholds"].update(p=0.5, s=0.5, latency_s=float(l_cal), m_max_mib=float(nc_tight))
        batch.probes.append(
            _request(
                context,
                spec,
                f"identity_s01_O11_core50_nc_q{nc_tight}",
                "identity_s01_stamp",
                "core50_nc",
                True,
                9,
            )
        )
        return batch
    if not _role_complete(evidence, "identity_s04_stamp", "core50_nc", 1):
        reserved = [0, 0, 0, 4 * 1024**2, 4 * 1024**2, 4 * 1024**2, 0, 0, 0]
        spec = stamp_dev(_template(root, "core50_nc"), role="identity_s04_stamp", method="R4", dataset="core50_nc")
        spec["training"]["eval_batch"] = nc_eval
        apply_quota(spec, nc_tight)
        spec["resource_envelope"] = {"enabled": True, "reserved_bytes_by_experience": reserved}
        spec["controller"].update(enabled=True, plugin_policy="adaptive")
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"].update(PLUGIN_DEFAULTS)
        spec["controller"]["thresholds"].update(latency_s=float(l_cal), m_max_mib=float(nc_tight))
        batch.probes.append(
            _request(context, spec, f"identity_s04_dyn_R4_core50_nc_q{nc_tight}", "identity_s04_stamp", "core50_nc", True, 9)
        )
        return batch
    if not (
        _role_complete(evidence, "io_off", "core50_nc", 1) and _role_complete(evidence, "io_on", "core50_nc", 1)
    ):
        for enabled, role in ((False, "io_off"), (True, "io_on")):
            spec = stamp_dev(_template(root, "core50_nc"), role=role, method=role, dataset="core50_nc")
            spec["training"]["eval_batch"] = int(eval_choice["core50_nc"])
            apply_quota(spec, nc_loose)
            spec["prefetch"].update(enabled=enabled, queue_depth=2, pin_memory=False, num_workers=0)
            spec["data_supply"] = {"record_hashes": True, "profile": "natural_ondemand"}
            batch.probes.append(_request(context, spec, f"{role}_core50_nc", role, "core50_nc", True, 9))
        return batch
    return batch


def planned_sensitivity_variants(evidence: dict[str, Any], dataset: str = "core50_nc") -> dict[str, dict[str, Any]]:
    """PLAN §5.2 one-factor probes. Baseline (Thr0=0.05, α=0.1, β=0.2, δ=0, lr=0.01, b16, r200, tight) is reused."""
    n_exp = N_EXPECTED[dataset]
    variants: dict[str, dict[str, Any]] = {
        "thr_half": {"controller": {"thr0": 0.025, "enabled": True, "plugin_policy": "adaptive"}},
        "thr_double": {"controller": {"thr0": 0.1, "enabled": True, "plugin_policy": "adaptive"}},
        "alpha_half": {
            "controller": {"enabled": True, "plugin_policy": "adaptive", "updates": {"alpha": 0.05, "beta": 0.2}}
        },
        "alpha_double": {
            "controller": {"enabled": True, "plugin_policy": "adaptive", "updates": {"alpha": 0.2, "beta": 0.2}}
        },
        "beta_half": {
            "controller": {"enabled": True, "plugin_policy": "adaptive", "updates": {"alpha": 0.1, "beta": 0.1}}
        },
        "beta_double": {
            "controller": {"enabled": True, "plugin_policy": "adaptive", "updates": {"alpha": 0.1, "beta": 0.4}}
        },
        "lr_half": {
            "training": {"optimizer": {"name": "sgd", "lr": 0.005, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"}}
        },
        "lr_double": {
            "training": {"optimizer": {"name": "sgd", "lr": 0.02, "momentum": 0.9, "weight_decay": 0.0, "scheduler": "none"}}
        },
        "delta_half_life": {
            "controller": {
                "enabled": True,
                "plugin_policy": "adaptive",
                "delta": float(math.log(2.0) / float(n_exp - 1)),
            }
        },
        "initial_batch32": {
            "training": {"new_batch": 32, "replay_batch": 32},
            "controller": {"mb0": 32.0, "enabled": True, "plugin_policy": "adaptive"},
        },
        "initial_replay1000": {
            "replay": {"capacity": 1000},
            "controller": {"mr0": 1000.0, "enabled": True, "plugin_policy": "adaptive"},
        },
        "quota_loose": {"_quota": "loose"},
    }
    q_tight = _chosen_tight(evidence, dataset)
    q_loose = _verified_loose(evidence, dataset, q_tight) if q_tight is not None else None
    if q_tight is not None and q_loose is not None:
        from orion_repro.stages.effectiveness_v3.calibration import CalibrationError, mid_quota_mib

        try:
            variants["quota_mid"] = {"_quota": "mid", "_quota_mib": int(mid_quota_mib(q_tight, q_loose))}
        except CalibrationError:
            pass
    return variants


def _dyn_r_mib(row: dict[str, Any]) -> int:
    reserved = row.get("reserved_bytes_by_experience") or []
    highs = [int(x) for x in reserved if int(x or 0) > 0]
    if not highs:
        return 0
    return int(max(highs) // (1024**2))


def _need_dyn_min_positive(evidence: dict[str, Any]) -> bool:
    probe = next(
        (r for r in evidence.get("runs") or [] if r.get("role") == "dyn_reservation_probe"),
        None,
    )
    if probe is None:
        return False
    if probe.get("status") == "completed" and probe.get("transition_first_ok"):
        return False
    if probe.get("status") not in {"cuda_oom", "host_oom", "budget_exceeded"}:
        return False
    if probe.get("failure_phase") not in {"resource_transition", "training"}:
        return False
    if _dyn_r_mib(probe) <= 4:
        return False
    return not _role_complete(evidence, "dyn_min_positive", "core50_nc", 1)


def _dyn_anchor(evidence: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        r
        for r in evidence.get("runs") or []
        if r.get("role") in {"dyn_reservation_probe", "dyn_min_positive"}
    ]
    ok = [r for r in rows if r.get("status") == "completed" and r.get("transition_first_ok")]
    if ok:
        return ok[-1]
    completed = [r for r in rows if r.get("status") == "completed"]
    return completed[-1] if completed else None


def _quota_rows(evidence: dict[str, Any], role: str, dataset: str) -> list[dict[str, Any]]:
    return [
        r
        for r in evidence.get("runs") or []
        if r.get("role") == role and r.get("dataset") == dataset
    ]


def _pending(rows: list[dict[str, Any]]) -> bool:
    return any(r.get("status") in {None, "not_run", "interrupted"} for r in rows)


def _next_quota_scan(evidence: dict[str, Any], dataset: str) -> tuple[list[int], str | None]:
    peak = _profile_reserved_mib(evidence, dataset)
    if peak is None:
        return [], f"no reserved peak for {dataset} S0 profile"
    rows = _quota_rows(evidence, "quota_scan", dataset)
    if _pending(rows):
        return [], f"quota_scan pending for {dataset}"
    start = max(16, int((int(peak) + 15) // 16 * 16))
    used = {int(r["quota_mib"]) for r in rows if r.get("quota_mib") not in (None, "")}
    if used:
        start = max(used) + 16
    completed = [
        r
        for r in rows
        if r.get("status") == "completed" and r.get("train_reserved_peak_ratio") not in (None, "")
    ]
    if completed:
        last = max(completed, key=lambda r: int(r["quota_mib"]))
        if float(last["train_reserved_peak_ratio"]) < 0.80:
            return [], (
                f"Q_tight not identifiable for {dataset}: reserved/quota fell below 0.80 "
                f"without a 80-95% grid point"
            )
    cap = max(int(peak) * 4, start + 16 * 24, 512)
    if start > cap:
        return [], f"Q_tight scan exhausted for {dataset} above {cap} MiB"
    quotas: list[int] = []
    quota = start
    while quota <= cap and len(quotas) < 8:
        if quota not in used:
            quotas.append(quota)
        quota += 16
    if not quotas:
        return [], f"Q_tight not identifiable for {dataset}"
    return quotas, None


def _verified_loose(evidence: dict[str, Any], dataset: str, q_tight: int) -> int | None:
    target = 2 * int(q_tight)
    ok = [
        int(r["quota_mib"])
        for r in _quota_rows(evidence, "q_loose_verify", dataset)
        if r.get("status") == "completed" and r.get("quota_mib") not in (None, "") and int(r["quota_mib"]) >= target
    ]
    return min(ok) if ok else None


def _next_loose_quota(evidence: dict[str, Any], dataset: str, q_tight: int) -> tuple[int, str | None]:
    rows = _quota_rows(evidence, "q_loose_verify", dataset)
    if _pending(rows):
        return 0, f"q_loose_verify pending for {dataset}"
    target = 2 * int(q_tight)
    used = {int(r["quota_mib"]) for r in rows if r.get("quota_mib") not in (None, "")}
    next_q = target if not used else max(used) + 16
    if next_q < target:
        next_q = target
    cap = max(8 * int(q_tight), target)
    if next_q > cap:
        return 0, f"Q_loose not feasible for {dataset} up to {cap} MiB"
    return int(next_q), None


def _profile_reserved_mib(evidence: dict[str, Any], dataset: str) -> float | None:
    peaks = []
    for row in evidence.get("runs") or []:
        if row.get("dataset") != dataset or row.get("role") != "s0_profile":
            continue
        if row.get("status") != "completed":
            continue
        ratio = row.get("train_reserved_peak_ratio")
        quota = row.get("quota_mib")
        if ratio not in (None, "") and quota:
            peaks.append(float(ratio) * float(quota))
    return max(peaks) if peaks else None


def _chosen_tight(evidence: dict[str, Any], dataset: str) -> int | None:
    rows = [
        r
        for r in evidence.get("runs") or []
        if r.get("role") == "quota_scan"
        and r.get("dataset") == dataset
        and r.get("status") == "completed"
        and r.get("train_reserved_peak_ratio") not in (None, "")
        and 0.80 <= float(r["train_reserved_peak_ratio"]) <= 0.95
    ]
    if not rows:
        return None
    return min(int(r["quota_mib"]) for r in rows)


def chosen_eval_batch(evidence: dict[str, Any]) -> dict[str, int | None]:
    out: dict[str, int | None] = {dataset: None for dataset in PRIMARY_DATASETS}
    for dataset in PRIMARY_DATASETS:
        rows = [
            r
            for r in evidence.get("runs") or []
            if r.get("role") == "eval_batch" and r.get("dataset") == dataset
        ]
        for batch in EVAL_BATCH_CANDIDATES:
            hits = [r for r in rows if int(r.get("eval_batch") or 0) == batch]
            if hits and all(h.get("status") == "completed" for h in hits):
                out[dataset] = int(batch)
                break
    return out


def _request(
    context: StageContext,
    spec: dict[str, Any],
    probe_id: str,
    role: str,
    dataset: str,
    full_stream: bool,
    n_expected: int,
) -> ProbeRequest:
    validate_mapping(spec, require_provenance=False)
    rel = f"configs/{STUDY_ID}/{context.revision}/dev/{probe_id}.yaml"
    spec["v3_probe_id"] = probe_id
    spec["v3_full_stream"] = full_stream
    spec["v3_n_expected"] = n_expected
    return ProbeRequest(
        probe_id=probe_id,
        role=role,
        dataset=dataset,
        full_stream=full_stream,
        n_expected=n_expected,
        spec=spec,
        rel_config=rel,
    )


def emit_probe_configs(batch: ProbeBatch, context: StageContext) -> list[str]:
    rels = []
    for probe in batch.probes:
        path = context.root / probe.rel_config
        context.assert_output_path(path)
        write_yaml(path, probe.spec)
        rels.append(probe.rel_config)
    payload = {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "executable": True,
        "phase": "development",
        "configs": rels,
        "probes": [
            {
                "probe_id": p.probe_id,
                "role": p.role,
                "dataset": p.dataset,
                "full_stream": p.full_stream,
                "n_expected": p.n_expected,
                "config": p.rel_config,
            }
            for p in batch.probes
        ],
        "blocked_reason": batch.blocked_reason,
    }
    matrix_path = context.revision_dir / "dev_batch.yaml"
    context.assert_output_path(matrix_path)
    write_yaml(matrix_path, payload)
    return rels


def collect_row(rel: str, role: str, context: StageContext, run_dir: Path | None) -> dict[str, Any]:
    spec = load_yaml(context.root / rel)
    resolved_path = run_dir / "resolved_config.yaml" if run_dir else None
    resolved = load_yaml(resolved_path) if resolved_path and resolved_path.is_file() else spec
    summary: dict[str, Any] = {}
    if run_dir and (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = _read_csv(run_dir / "experience_metrics.csv") if run_dir else []
    phases = _read_csv(run_dir / "phase_trace.csv") if run_dir else []
    control = []
    if run_dir and (run_dir / "control_trace.jsonl").is_file():
        control = [
            json.loads(line)
            for line in (run_dir / "control_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    train_phases = [p for p in phases if p.get("phase") == "training"]
    quota = spec["budget"].get("limit_bytes")
    ratios = []
    for phase in train_phases:
        reserved = phase.get("reserved_peak_bytes")
        if reserved not in (None, "", "None") and quota:
            ratios.append(float(reserved) / float(quota))
    learning = [float(m["learning_s"]) for m in metrics if m.get("learning_s") not in (None, "")]
    produce = [float(m.get("prefetch_produce_s") or 0) for m in metrics]
    wait = [float(m.get("prefetch_wait_s") or 0) for m in metrics]
    supply = None
    if learning:
        blocked = sum(wait) if spec["prefetch"]["enabled"] else sum(produce)
        supply = blocked / max(sum(learning), 1e-9)
    changed = False
    for item in control:
        if item.get("applied_new_batch") not in (None, spec["training"]["new_batch"]):
            changed = True
        if item.get("applied_replay") not in (None, spec["replay"]["capacity"]):
            changed = True
        if item.get("applied_optimizer_mode") == "advanced":
            changed = True
    n_expected = int(spec.get("v3_n_expected") or spec["dataset"]["experience_limit"])
    n_trained = summary.get("n_experiences_trained")
    n_evaluated = summary.get("n_experiences_evaluated")
    if n_trained is None:
        n_trained = len(metrics) if summary.get("status") == "completed" else summary.get("n_experiences_run")
    artifacts = {}
    if run_dir:
        for name in (
            "summary.json",
            "phase_trace.csv",
            "experience_metrics.csv",
            "resolved_config.yaml",
            "control_trace.jsonl",
            "events.jsonl",
            "plugin_activity.jsonl",
            "resource_trace.csv",
        ):
            path = run_dir / name
            if path.is_file():
                artifacts[name] = sha256_file(path)
    actions: list[dict[str, Any]] = []
    for item in control:
        last_only = item.get("reason") == "last_experience" or (
            item.get("applied_new_batch") is None and item.get("controller") != "skipped"
        )
        action = {
            "t": item.get("t"),
            "applied_new_batch": item.get("applied_new_batch"),
            "applied_replay": item.get("applied_replay"),
            "applied_optimizer_mode": item.get("applied_optimizer_mode"),
            "last_experience_only": bool(last_only),
        }
        actions.append(action)
    cpu_mean = None
    if run_dir:
        resources = _read_csv(run_dir / "resource_trace.csv")
        cpu_vals = [float(r["cpu_percent"]) for r in resources if r.get("cpu_percent") not in (None, "")]
        if cpu_vals:
            cpu_mean = sum(cpu_vals) / len(cpu_vals)
    row = {
        "probe_id": spec.get("v3_probe_id") or Path(rel).stem,
        "role": role,
        "dataset": spec["dataset"]["name"],
        "config": rel,
        "config_id": Path(rel).stem,
        "config_hash": canonical_hash(spec),
        "run_id": summary.get("run_id"),
        "status": summary.get("status") or "not_run",
        "feedback_source": spec["controller"]["feedback_source"],
        "controller_feedback_source": spec["controller"]["feedback_source"],
        "eval_batch": spec["training"]["eval_batch"],
        "quota_mib": int(spec["budget"]["limit_bytes"] / (1024**2)) if spec["budget"].get("limit_bytes") else None,
        "new_batch": spec["training"]["new_batch"],
        "replay_capacity": spec["replay"]["capacity"],
        "p_diag": summary.get("p_diag"),
        "s_initial": summary.get("s_initial"),
        "online_total_s": summary.get("online_total_s"),
        "learning_s_by_experience": learning,
        "train_reserved_peak_ratio": max(ratios) if ratios else None,
        "integer_config_changed": changed,
        "applied_actions": actions,
        "supply_wait_ratio": supply,
        "prefetch_produce_s": sum(produce) if produce else 0.0,
        "prefetch_wait_s": sum(wait) if wait else 0.0,
        "cpu_percent_mean": cpu_mean,
        "full_stream": bool(spec.get("v3_full_stream", n_expected == N_EXPECTED.get(spec["dataset"]["name"]))),
        "n_expected": n_expected,
        "n_trained": n_trained,
        "n_evaluated": n_evaluated,
        "failure_phase": summary.get("failure_phase"),
        "artifact_hashes": artifacts,
        "source_hash": resolved.get("code_revision_or_snapshot"),
        "environment_hash": resolved.get("environment_lock_sha256"),
        "split_hash": resolved.get("dataset_manifest_sha256"),
        "phase": spec.get("phase"),
        "plugin_policy": spec["controller"].get("plugin_policy"),
        "optional_plugins": spec["algorithm"].get("optional_plugins"),
        "reserved_bytes_by_experience": (spec.get("resource_envelope") or {}).get("reserved_bytes_by_experience"),
        "transition_first_ok": _transition_ok(run_dir, spec, summary),
    }
    return row


def empty_manifest(revision: str) -> dict[str, Any]:
    return {"kind": "probe_manifest", "study_id": STUDY_ID, "revision": revision, "runs": []}


def record_host_capability(evidence: dict[str, Any]) -> dict[str, Any]:
    from orion_repro.memory.host_enforcement import probe_delegated_cgroup, probe_host_enforcement

    if any(r.get("role") == "host_capability" for r in evidence.get("runs") or []):
        return evidence
    probe = probe_host_enforcement()
    delegated = probe_delegated_cgroup(None)
    available = bool(delegated.get("available"))
    evidence.setdefault("runs", []).append(
        {
            "probe_id": "host_capability",
            "role": "host_capability",
            "dataset": "core50_nc",
            "status": "available" if available else "unavailable",
            "feedback_source": "development_val_seen",
            "full_stream": False,
            "n_expected": 0,
            "n_trained": 0,
            "n_evaluated": 0,
            "host_probe": probe,
            "delegated": delegated,
        }
    )
    return evidence


def ingest_batch(batch: ProbeBatch, context: StageContext, evidence: dict[str, Any]) -> dict[str, Any]:
    progress = {"results": []}
    if context.progress_path.exists():
        progress = json.loads(context.progress_path.read_text(encoding="utf-8"))
    by_config = {}
    for row in progress.get("results") or []:
        cfg = row.get("config")
        if cfg:
            by_config[cfg] = row
    for probe in batch.probes:
        match = by_config.get(probe.rel_config)
        run_dir = None
        if match and match.get("run_id"):
            run_dir = context.root / "runs" / match["run_id"]
        collected = collect_row(probe.rel_config, probe.role, context, run_dir)
        if probe.role in {"dyn_reservation_probe", "dyn_min_positive", "identity_s04_stamp"} and collected.get("run_id"):
            from orion_repro.stages.effectiveness_v3.evidence import dyn_window_from_run

            collected["dyn_window"] = dyn_window_from_run(collected, context)
        evidence.setdefault("attempts", []).append(copy.deepcopy(collected))
        evidence["runs"] = [r for r in evidence.get("runs") or [] if r.get("probe_id") != probe.probe_id]
        evidence.setdefault("runs", []).append(collected)
    from orion_repro.stages.effectiveness_v3.evidence import (
        audit_dyn_windows,
        factor_diagnostics,
        write_evidence_closure,
    )

    write_evidence_closure(evidence, context)
    factor_diagnostics(evidence, context)
    audit_dyn_windows(evidence, context)
    return evidence


def _transition_ok(run_dir: Path | None, spec: dict[str, Any], summary: dict[str, Any]) -> bool:
    if not (spec.get("resource_envelope") or {}).get("enabled"):
        return True
    trans_ok = True
    if run_dir and (run_dir / "events.jsonl").is_file():
        events = []
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("phase") == "resource_transition":
                events.append(event)
        first_high = next((e for e in events if int(e.get("experience", -1)) == 3), None)
        trans_ok = bool(first_high and first_high.get("status") == "ok")
    if summary.get("failure_phase") == "resource_transition":
        trans_ok = False
    return trans_ok


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
