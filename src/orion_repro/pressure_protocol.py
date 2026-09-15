"""pressure_v2 design validation, calibration, freeze, and 54-cell emission."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from orion_repro.runner.spec import load_yaml, validate_mapping
from orion_repro.control.urge import preference_weights, coefficients_from_weights

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "pressure_v2_protocol_v1"
PROTOCOL_ID = "pressure_v2_paper_feedback_v1"
SEEDS = (0, 1, 2)
TIGHT_METHODS = ("S0", "S_star", "O00", "O10", "O01", "O11", "R11")
LOOSE_METHODS = ("S0", "S_star", "O11")
DYN_METHODS = ("S0", "S_star_dynamic", "O11")
DECAY_METHODS = ("O11_half_life",)
PREF_METHODS = ("O11_latency", "O11_ps")
IO_METHODS = ("static_off", "static_on")
STATIC_GRID = [(16, 200), (16, 2000), (64, 200), (64, 2000), (256, 200), (256, 2000)]
PLUGIN_DEFAULTS = {
    "patterns_per_exp": 50,
    "memory_strength": 0.5,
    "ewc_lambda": 100.0,
}


class DesignError(ValueError):
    pass


class CalibrationError(ValueError):
    pass


def _sha(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    if design.get("kind") != "experiment_design":
        raise DesignError("only kind=experiment_design is accepted")
    if design.get("study_id") != "pressure_v2":
        raise DesignError("design study_id must be pressure_v2")
    if design.get("executable") is True:
        raise DesignError("design files are not executable training matrices")
    if int(design.get("formal_count", 0)) != 54:
        raise DesignError("formal_count must be 54")
    return design


def _require_dev_feedback(row: dict[str, Any]) -> None:
    source = row.get("feedback_source") or row.get("controller_feedback_source")
    if source and source != "development_val_seen":
        raise CalibrationError(f"non-development feedback rejected: {source}")


def _median(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise CalibrationError("empty median")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _select_static(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [c for c in candidates if c.get("status") == "completed"]
    failed = [c for c in candidates if c.get("status") in {"cuda_oom", "host_oom", "budget_exceeded"}]
    errors = [c for c in candidates if c.get("status") == "implementation_error"]
    if errors:
        raise CalibrationError("implementation errors must be fixed, not treated as resource infeasibility")
    if not feasible:
        raise CalibrationError("no feasible static candidates")
    ranked = sorted(
        feasible,
        key=lambda row: (
            -((float(row["p_diag"]) + float(row["s_initial"])) / 2.0),
            float(row["online_total_s"]),
            str(row.get("config_id") or row.get("config") or ""),
        ),
    )
    winner = ranked[0]
    return {
        "new_batch": int(winner["new_batch"]),
        "replay_capacity": int(winner["replay_capacity"]),
        "config_id": str(winner.get("config_id") or winner.get("config")),
        "run_id": winner.get("run_id"),
        "score": (float(winner["p_diag"]) + float(winner["s_initial"])) / 2.0,
        "n_feasible": len(feasible),
        "n_resource_failed": len(failed),
        "rule": "max (P+S)/2; then shorter online_total_s; then lexical config_id",
    }


def _eval_batch_from_probes(rows: list[dict[str, Any]]) -> int:
    eval_rows = [r for r in rows if r.get("role") == "d0_eval_batch"]
    if not eval_rows:
        raise CalibrationError("missing d0_eval_batch evidence")
    for batch in (32, 8):
        hits = [r for r in eval_rows if int(r.get("eval_batch", 0)) == batch]
        if hits and all(h.get("status") == "completed" for h in hits):
            return batch
    raise CalibrationError("eval_batch 32 and 8 still fail; evaluation bottleneck not separated")


def _quota_choice(rows: list[dict[str, Any]]) -> tuple[int, int, dict[str, Any]]:
    scans = [r for r in rows if r.get("role") == "quota_scan"]
    if not scans:
        raise CalibrationError("missing quota_scan evidence")
    evidence = []
    chosen = None
    for row in sorted(scans, key=lambda r: int(r["quota_mib"])):
        status = row.get("status")
        ratio = row.get("train_reserved_peak_ratio")
        item = {
            "quota_mib": int(row["quota_mib"]),
            "status": status,
            "train_reserved_peak_ratio": ratio,
            "run_id": row.get("run_id"),
        }
        evidence.append(item)
        if status != "completed" or ratio is None:
            continue
        if 0.80 <= float(ratio) <= 0.95 and chosen is None:
            chosen = row
    if chosen is None:
        raise CalibrationError("no Q_tight with completed b16 train/eval and reserved peak in 80-95% of quota")
    q_tight = int(chosen["quota_mib"])
    loose_rows = [r for r in rows if r.get("role") == "q_loose_verify"]
    q_loose = 2 * q_tight
    if loose_rows:
        ok = [r for r in loose_rows if r.get("status") == "completed" and int(r["quota_mib"]) == q_loose]
        if not ok:
            completed = [r for r in loose_rows if r.get("status") == "completed"]
            if not completed:
                raise CalibrationError("Q_loose=2*Q_tight was not feasible")
            q_loose = int(completed[0]["quota_mib"])
    return q_tight, q_loose, {"scans": evidence, "selected_tight_run": chosen.get("run_id")}


def calibrate(probe_manifest: dict[str, Any]) -> dict[str, Any]:
    if probe_manifest.get("study_id") != "pressure_v2":
        raise CalibrationError("probe_manifest study_id must be pressure_v2")
    rows = list(probe_manifest.get("runs") or [])
    if not rows:
        raise CalibrationError("probe_manifest has no runs")
    for row in rows:
        if row.get("role") in {
            "static_search_tight",
            "static_search_loose",
            "static_search_dyn",
            "l_cal",
            "control_2x2",
        }:
            _require_dev_feedback(row)
        if row.get("status") is None:
            raise CalibrationError(f"candidate missing outcome: {row.get('config_id')}")
        if row.get("status") == "implementation_error":
            raise CalibrationError(f"implementation error in {row.get('config_id')}")
    eval_batch = _eval_batch_from_probes(rows)
    q_tight, q_loose, quota_evidence = _quota_choice(rows)
    l_rows = [r for r in rows if r.get("role") == "l_cal" and r.get("status") == "completed"]
    if len(l_rows) < 2:
        raise CalibrationError("L_cal requires two completed Q_loose static b16/r200 runs")
    learning = []
    for row in l_rows:
        values = row.get("learning_s_by_experience")
        if not values:
            raise CalibrationError("L_cal run missing per-experience learning_s")
        learning.extend(float(v) for v in values)
    l_cal = _median(learning)
    s_tight = _select_static([r for r in rows if r.get("role") == "static_search_tight"])
    s_loose = _select_static([r for r in rows if r.get("role") == "static_search_loose"])
    dyn_search = [r for r in rows if r.get("role") == "static_search_dyn"]
    s_dyn = _select_static(dyn_search) if dyn_search else copy.deepcopy(s_tight)
    control = [r for r in rows if r.get("role") == "control_2x2"]
    identifiable = any(
        r.get("status") == "completed" and r.get("integer_config_changed") for r in control
    )
    dyn_probe = [r for r in rows if r.get("role") == "dyn_reservation_probe"]
    reserved = [0] * 9
    dyn_quota = q_tight
    dyn_constructed = False
    if dyn_probe:
        ok = next((r for r in dyn_probe if r.get("status") == "completed" and r.get("transition_first_ok")), None)
        if ok:
            reserved = [int(x) for x in ok["reserved_bytes_by_experience"]]
            dyn_quota = int(ok.get("quota_mib", q_tight)) * 1024**2 if ok.get("quota_mib") else q_tight * 1024**2
            dyn_constructed = True
        else:
            dyn_quota = q_tight * 1024**2
    else:
        dyn_quota = q_tight * 1024**2
    io_off = next((r for r in rows if r.get("role") == "io_off"), None)
    io_on = next((r for r in rows if r.get("role") == "io_on"), None)
    io_ratio = None
    io_constructed = False
    if io_off and io_off.get("status") == "completed":
        io_ratio = io_off.get("supply_wait_ratio")
        io_constructed = io_ratio is not None and float(io_ratio) >= 0.10
    coverage = {
        "eval_separation": "realized",
        "pressure_interval": "realized",
        "latency_calibration": "realized",
        "static_search": "realized",
        "control_identifiable": "realized" if identifiable else "failed_to_construct",
        "dynamic_reservation": "realized" if dyn_constructed else "failed_to_construct",
        "io_prefetch": "realized" if io_constructed else "failed_to_construct",
        "preferences": "realized" if identifiable else "failed_to_construct",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_batch": int(eval_batch),
        "q_tight_mib": int(q_tight),
        "q_loose_mib": int(q_loose),
        "quota_bytes_tight": int(q_tight) * 1024**2,
        "quota_bytes_loose": int(q_loose) * 1024**2,
        "l_cal_s": float(l_cal),
        "thr0": 0.05,
        "delta": 0.0,
        "alpha": 0.1,
        "beta": 0.2,
        "initial_new_batch": 16,
        "initial_replay": 200,
        "optional_plugins": "gem_ewc",
        "optional_start_enabled": False,
        "m_batch": 1.0,
        "m_frame": 1.0,
        "cost_model_note": "count-space proxy m_batch=m_frame=1.0; not physical MiB",
        "static_selections": {"tight": s_tight, "loose": s_loose, "dyn": s_dyn},
        "reserved_bytes_by_experience": reserved,
        "dyn_quota_bytes": int(dyn_quota) if dyn_constructed else int(q_tight) * 1024**2,
        "quota_evidence": quota_evidence,
        "control_identifiable": bool(identifiable),
        "scenario_coverage": coverage,
        "io_prefetch": {
            "constructed": bool(io_constructed),
            "wait_ratio": float(io_ratio) if io_ratio is not None else 0.0,
            "pin_memory": False,
            "queue_depth": 2,
            "num_workers": 0,
        },
        "source_runs": [r.get("run_id") for r in rows if r.get("run_id")],
        "probe_manifest_hash": _sha(probe_manifest),
    }


def interleave_methods(methods: tuple[str, ...] | list[str], seeds: tuple[int, ...] = SEEDS) -> list[dict[str, Any]]:
    methods = list(methods)
    order = []
    for i, seed in enumerate(seeds):
        rotated = methods[i % len(methods) :] + methods[: i % len(methods)]
        for method in rotated:
            order.append({"seed": int(seed), "method": method})
    return order


def freeze(calibration: dict[str, Any], source_identity: dict[str, Any]) -> dict[str, Any]:
    required = [
        "eval_batch",
        "quota_bytes_tight",
        "quota_bytes_loose",
        "l_cal_s",
        "static_selections",
        "reserved_bytes_by_experience",
        "scenario_coverage",
    ]
    missing = [k for k in required if calibration.get(k) in (None, "TBD", "tbd")]
    if missing:
        raise CalibrationError(f"refusing to freeze unresolved fields: {missing}")
    run_order = (
        [{"group": "tight", **row} for row in interleave_methods(TIGHT_METHODS)]
        + [{"group": "loose", **row} for row in interleave_methods(LOOSE_METHODS)]
        + [{"group": "dyn", **row} for row in interleave_methods(DYN_METHODS)]
        + [{"group": "decay", **row} for row in interleave_methods(DECAY_METHODS)]
        + [{"group": "pref", **row} for row in interleave_methods(PREF_METHODS)]
        + [{"group": "io", **row} for row in interleave_methods(IO_METHODS)]
    )
    if len(run_order) != 54:
        raise CalibrationError(f"run_order length {len(run_order)} != 54")
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_id": "pressure_v2",
        "source_identity": source_identity,
        "dataset_hash": source_identity.get("dataset_hash"),
        "source_hash": source_identity.get("source_hash"),
        "design_hash": source_identity.get("design_hash"),
        "calibration_hash": calibration.get("probe_manifest_hash"),
        "quota_bytes_tight": int(calibration["quota_bytes_tight"]),
        "quota_bytes_loose": int(calibration["quota_bytes_loose"]),
        "eval_batch": int(calibration["eval_batch"]),
        "l_cal_s": float(calibration["l_cal_s"]),
        "latency_threshold_s": float(calibration["l_cal_s"]),
        "memory_threshold_mib_tight": float(calibration["q_tight_mib"]),
        "memory_threshold_mib_loose": float(calibration["q_loose_mib"]),
        "q_tight_mib": int(calibration["q_tight_mib"]),
        "q_loose_mib": int(calibration["q_loose_mib"]),
        "thr0": 0.05,
        "delta": 0.0,
        "alpha": 0.1,
        "beta": 0.2,
        "initial_counts": {"new_batch": 16, "replay_capacity": 200},
        "optional_plugins": "gem_ewc",
        "optional_start_enabled": False,
        "m_batch": 1.0,
        "m_frame": 1.0,
        "cost_model_note": calibration.get("cost_model_note"),
        "static_selections": calibration["static_selections"],
        "reserved_bytes_by_experience": [int(x) for x in calibration["reserved_bytes_by_experience"]],
        "dyn_quota_bytes": int(calibration["dyn_quota_bytes"]),
        "seeds": list(SEEDS),
        "run_order": run_order,
        "calibration_evidence": {
            "source_runs": calibration.get("source_runs"),
            "quota": calibration.get("quota_evidence"),
            "control_identifiable": calibration.get("control_identifiable"),
        },
        "scenario_coverage": calibration["scenario_coverage"],
        "io_prefetch": calibration["io_prefetch"],
        "plugin_defaults": PLUGIN_DEFAULTS,
        "decay_delta": math.log(2.0) / 8.0,
        "preference_orders": {
            "latency": ["latency", "memory", "plasticity", "stability"],
            "ps": ["plasticity", "stability", "memory", "latency"],
        },
    }
    nulls = [k for k, v in frozen.items() if v is None]
    if nulls:
        raise CalibrationError(f"frozen protocol has null fields: {nulls}")
    frozen["frozen_hash"] = _sha({k: v for k, v in frozen.items() if k != "frozen_hash"})
    return frozen


def _write_yaml(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _base_formal(root: Path, seed: int) -> dict[str, Any]:
    spec = load_yaml(ROOT / "configs/formal/e03/core50_nc/orion_formula/seed0.yaml")
    spec.update(
        study_id="pressure_v2",
        reuse_version=3,
        protocol_id=PROTOCOL_ID,
        phase="formal",
        alignment_version="pressure_v2",
    )
    spec["seeds"] = dict.fromkeys(("model", "stream", "replay", "augmentation"), int(seed))
    spec["algorithm"].update(PLUGIN_DEFAULTS)
    spec["controller"]["m_batch"] = 1.0
    spec["controller"]["m_frame"] = 1.0
    spec["budget"].update(
        enforcement="device_allocator_enforced",
        controlled_resource="device",
        guard_mode="stop",
    )
    spec["prefetch"].update(enabled=False, queue_depth=1, pin_memory=False, num_workers=0)
    spec["resource_envelope"] = {
        "enabled": False,
        "reserved_bytes_by_experience": [0] * 9,
    }
    spec["data_supply"] = {"record_hashes": False, "profile": "default"}
    return spec


def _apply_orion(spec: dict[str, Any], *, plugins: bool, policy: str, enabled: bool) -> None:
    spec["controller"]["enabled"] = bool(enabled)
    spec["controller"]["plugin_policy"] = policy
    if plugins:
        spec["algorithm"]["optional_plugins"] = "gem_ewc"
        spec["algorithm"]["optional_start_enabled"] = False
        spec["algorithm"].update(PLUGIN_DEFAULTS)
    else:
        spec["algorithm"]["optional_plugins"] = "none"
        spec["algorithm"]["optional_start_enabled"] = False


def _cell_spec(frozen: dict[str, Any], group: str, method: str, seed: int, root: Path) -> dict[str, Any]:
    spec = _base_formal(root, seed)
    spec["eval_batch_note"] = "frozen"
    spec["training"]["eval_batch"] = int(frozen["eval_batch"])
    spec["frozen_protocol_path"] = "experiments/pressure_v2/frozen_protocol.json"
    spec["frozen_hash"] = frozen["frozen_hash"]
    spec["controller"]["thr0"] = frozen["thr0"]
    spec["controller"]["delta"] = frozen["delta"]
    spec["controller"]["updates"] = {"alpha": frozen["alpha"], "beta": frozen["beta"]}
    spec["controller"]["mb0"] = float(frozen["initial_counts"]["new_batch"])
    spec["controller"]["mr0"] = float(frozen["initial_counts"]["replay_capacity"])
    spec["training"]["new_batch"] = frozen["initial_counts"]["new_batch"]
    spec["training"]["replay_batch"] = frozen["initial_counts"]["new_batch"]
    spec["replay"]["capacity"] = frozen["initial_counts"]["replay_capacity"]
    spec["controller"]["feedback_source"] = "official_test_seen"
    spec["pressure_group"] = group
    spec["method_id"] = method

    def set_quota(quota_bytes: int, m_max_mib: float) -> None:
        spec["budget"]["limit_bytes"] = int(quota_bytes)
        spec["controller"]["thresholds"]["m_max_mib"] = float(m_max_mib)

    def set_latency(seconds: float) -> None:
        spec["controller"]["thresholds"]["latency_s"] = float(seconds)

    if group == "tight":
        spec["experiment_id"] = "P2_tight"
        spec["claim_ids"] = ["P2_tight"]
        set_quota(frozen["quota_bytes_tight"], frozen["q_tight_mib"])
        if method == "S0":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
        elif method == "S_star":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
            chosen = frozen["static_selections"]["tight"]
            spec["training"]["new_batch"] = chosen["new_batch"]
            spec["training"]["replay_batch"] = chosen["new_batch"]
            spec["replay"]["capacity"] = chosen["replay_capacity"]
        elif method == "O00":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
            set_latency(30.0)
            spec["controller"]["thresholds"]["m_max_mib"] = 4096.0
        elif method == "O10":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
            set_latency(frozen["l_cal_s"])
            spec["controller"]["thresholds"]["m_max_mib"] = 4096.0
        elif method == "O01":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
            set_latency(30.0)
        elif method == "O11":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
            set_latency(frozen["l_cal_s"])
        elif method == "R11":
            _apply_orion(spec, plugins=True, policy="fixed_default", enabled=True)
            set_latency(frozen["l_cal_s"])
        else:
            raise CalibrationError(method)
    elif group == "loose":
        spec["experiment_id"] = "P2_loose"
        spec["claim_ids"] = ["P2_loose"]
        set_quota(frozen["quota_bytes_loose"], frozen["q_loose_mib"])
        if method == "S0":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
        elif method == "S_star":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
            chosen = frozen["static_selections"]["loose"]
            spec["training"]["new_batch"] = chosen["new_batch"]
            spec["training"]["replay_batch"] = chosen["new_batch"]
            spec["replay"]["capacity"] = chosen["replay_capacity"]
        elif method == "O11":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
            set_latency(frozen["l_cal_s"])
        else:
            raise CalibrationError(method)
    elif group == "dyn":
        spec["experiment_id"] = "P2_dyn"
        spec["claim_ids"] = ["P2_dyn"]
        set_quota(frozen["dyn_quota_bytes"], frozen["dyn_quota_bytes"] / (1024**2))
        spec["resource_envelope"] = {
            "enabled": True,
            "schedule_path": "experiments/pressure_v2/dyn_schedule.json",
            "reserved_bytes_by_experience": frozen["reserved_bytes_by_experience"],
        }
        set_latency(frozen["l_cal_s"])
        if method == "S0":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
            set_latency(30.0)
        elif method == "S_star_dynamic":
            _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
            chosen = frozen["static_selections"]["dyn"]
            spec["training"]["new_batch"] = chosen["new_batch"]
            spec["training"]["replay_batch"] = chosen["new_batch"]
            spec["replay"]["capacity"] = chosen["replay_capacity"]
        elif method == "O11":
            _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        else:
            raise CalibrationError(method)
    elif group == "decay":
        spec["experiment_id"] = "P2_decay"
        spec["claim_ids"] = ["P2_decay"]
        set_quota(frozen["dyn_quota_bytes"], frozen["dyn_quota_bytes"] / (1024**2))
        spec["resource_envelope"] = {
            "enabled": True,
            "schedule_path": "experiments/pressure_v2/dyn_schedule.json",
            "reserved_bytes_by_experience": frozen["reserved_bytes_by_experience"],
        }
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        set_latency(frozen["l_cal_s"])
        spec["controller"]["delta"] = float(frozen["decay_delta"])
    elif group == "pref":
        spec["experiment_id"] = "P2_pref"
        spec["claim_ids"] = ["P2_pref"]
        set_quota(frozen["quota_bytes_tight"], frozen["q_tight_mib"])
        _apply_orion(spec, plugins=True, policy="adaptive", enabled=True)
        set_latency(frozen["l_cal_s"])
        order_key = "latency" if method == "O11_latency" else "ps"
        order = frozen["preference_orders"][order_key]
        spec["controller"]["preference"] = "ranked"
        spec["controller"]["preference_order"] = order
        spec["controller"]["coefficients"] = coefficients_from_weights(preference_weights(order))
    elif group == "io":
        spec["experiment_id"] = "P2_io"
        spec["claim_ids"] = ["P2_io"]
        set_quota(frozen["quota_bytes_loose"], frozen["q_loose_mib"])
        _apply_orion(spec, plugins=False, policy="adaptive", enabled=False)
        spec["data_supply"] = {
            "record_hashes": True,
            "profile": "natural_ondemand",
            "profile_path": "experiments/pressure_v2/io_profile.json",
        }
        spec["training"]["deterministic_algorithms"] = True
        on = method == "static_on"
        spec["prefetch"].update(
            enabled=on,
            queue_depth=int(frozen["io_prefetch"]["queue_depth"]),
            pin_memory=bool(frozen["io_prefetch"]["pin_memory"]),
            num_workers=int(frozen["io_prefetch"]["num_workers"]),
        )
    else:
        raise CalibrationError(group)
    validate_mapping(spec, require_provenance=False)
    return spec


def emit(frozen: dict[str, Any], root: Path = ROOT) -> list[str]:
    rels = []
    for item in frozen["run_order"]:
        spec = _cell_spec(frozen, item["group"], item["method"], item["seed"], root)
        rel = f"configs/pressure_v2/formal/{item['group']}/{item['method']}_s{item['seed']}.yaml"
        _write_yaml(root / rel, spec)
        rels.append(rel)
    if len(rels) != 54:
        raise CalibrationError(f"emitted {len(rels)} != 54")
    matrix = {
        "study_id": "pressure_v2",
        "name": "formal_54",
        "frozen_hash": frozen["frozen_hash"],
        "frozen_protocol_path": "experiments/pressure_v2/frozen_protocol.json",
        "configs": rels,
        "executable": True,
    }
    _write_yaml(root / "experiments/pressure_v2/formal.yaml", matrix)
    schedule = {
        "reserved_bytes_by_experience": frozen["reserved_bytes_by_experience"],
        "note": "same-process CUDA uint8 reservation; not model or replay bytes",
    }
    (root / "experiments/pressure_v2/dyn_schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    io_profile = {
        "profile": "natural_ondemand",
        "pin_memory": frozen["io_prefetch"]["pin_memory"],
        "queue_depth": frozen["io_prefetch"]["queue_depth"],
        "num_workers": frozen["io_prefetch"]["num_workers"],
        "constructed": frozen["io_prefetch"]["constructed"],
    }
    (root / "experiments/pressure_v2/io_profile.json").write_text(
        json.dumps(io_profile, indent=2), encoding="utf-8"
    )
    return rels
