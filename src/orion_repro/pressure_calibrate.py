"""D0/D1 development diagnostics, freeze, and formal emission for pressure_v2."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from orion_repro.pressure_protocol import (
    PLUGIN_DEFAULTS,
    STATIC_GRID,
    calibrate,
    emit,
    freeze,
    validate_design,
)
from orion_repro.pressure_study import PROGRESS, reconcile_active, run_config_list, save
from orion_repro.provenance import sha256_file, snapshot_source_tree
from orion_repro.runner.locks import acquire_locks
from orion_repro.runner.reuse import reuse_identity
from orion_repro.runner.spec import load_yaml, validate_mapping
from orion_repro.memory.resource_envelope import default_dyn_schedule

ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, spec: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return str(path.relative_to(ROOT))


def _stamp(spec: dict[str, Any], *, role: str, method: str) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec.update(
        study_id="pressure_v2",
        reuse_version=3,
        protocol_id="pressure_v2_development_v1",
        phase="development",
        alignment_version="pressure_v2",
        experiment_id=f"P2_{role}",
        claim_ids=[f"P2_{role}"],
        method_id=method,
        pressure_role=role,
    )
    return spec


def _core_dev() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/development/core50_nc_er_static.yaml")


def _cifar_dev() -> dict[str, Any]:
    return load_yaml(ROOT / "configs/development/cifar100_er_static.yaml")


def emit_matrix(name: str, rels: list[str]) -> Path:
    path = ROOT / f"experiments/pressure_v2/{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {"study_id": "pressure_v2", "name": name, "executable": True, "configs": rels},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def apply_quota(spec: dict[str, Any], quota_mib: int) -> None:
    spec["budget"].update(
        enforcement="device_allocator_enforced",
        controlled_resource="device",
        limit_bytes=int(quota_mib) * 1024**2,
    )
    spec["controller"]["thresholds"]["m_max_mib"] = float(quota_mib)


def emit_d0_eval() -> list[str]:
    rels = []
    for batch in (128, 32, 8):
        spec = _stamp(_cifar_dev(), role="d0_eval_batch", method=f"eval{batch}")
        spec["training"]["eval_batch"] = batch
        spec["dataset"]["experience_limit"] = 1
        apply_quota(spec, 128)
        rels.append(_write_yaml(ROOT / f"configs/pressure_v2/dev/d0_eval_b{batch}.yaml", spec))
    emit_matrix("d0_eval", rels)
    return rels


def emit_static_dev(
    *,
    role: str,
    quota_mib: int,
    batch: int,
    replay: int,
    eval_batch: int,
    plugins: str = "none",
    start_enabled: bool = False,
    controller: bool = False,
    seed: int = 0,
    extra: dict[str, Any] | None = None,
    name: str,
) -> str:
    spec = _stamp(_core_dev(), role=role, method=name)
    spec["seeds"] = dict.fromkeys(("model", "stream", "replay", "augmentation"), int(seed))
    spec["training"].update(new_batch=batch, replay_batch=batch, eval_batch=eval_batch)
    spec["replay"]["capacity"] = replay
    spec["algorithm"]["optional_plugins"] = plugins
    spec["algorithm"]["optional_start_enabled"] = bool(start_enabled)
    if plugins == "gem_ewc":
        spec["algorithm"].update(PLUGIN_DEFAULTS)
    spec["controller"]["enabled"] = bool(controller)
    apply_quota(spec, quota_mib)
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(spec.get(key), dict):
                spec[key].update(value)
            else:
                spec[key] = value
    validate_mapping(spec, require_provenance=False)
    return _write_yaml(ROOT / f"configs/pressure_v2/dev/{name}.yaml", spec)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_run_for_config(rel: str) -> Path | None:
    if not PROGRESS.exists():
        return None
    state = json.loads(PROGRESS.read_text())
    for row in reversed(state.get("results", [])):
        if row.get("config") in {rel, str(ROOT / rel)} and row.get("run_id"):
            return ROOT / "runs" / row["run_id"]
    return None


def collect_row(rel: str, role: str) -> dict[str, Any]:
    spec = load_yaml(ROOT / rel)
    run_dir = latest_run_for_config(rel)
    summary = {}
    if run_dir and (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text())
    metrics = read_csv(run_dir / "experience_metrics.csv") if run_dir else []
    phases = read_csv(run_dir / "phase_trace.csv") if run_dir else []
    control = []
    if run_dir and (run_dir / "control_trace.jsonl").is_file():
        control = [
            json.loads(line)
            for line in (run_dir / "control_trace.jsonl").read_text().splitlines()
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
    changed = False
    for item in control:
        if item.get("applied_new_batch") not in (None, spec["training"]["new_batch"]):
            changed = True
        if item.get("applied_replay") not in (None, spec["replay"]["capacity"]):
            changed = True
        if item.get("applied_optimizer_mode") == "advanced":
            changed = True
    produce = [float(m.get("prefetch_produce_s") or 0) for m in metrics]
    wait = [float(m.get("prefetch_wait_s") or 0) for m in metrics]
    supply = None
    if learning:
        # Producer work overlaps consumer waiting/training when queued.
        # Serial loading blocks training directly; queued producer work does not.
        blocked = sum(wait) if spec["prefetch"]["enabled"] else sum(produce)
        supply = blocked / max(sum(learning), 1e-9)
    trans_ok = True
    if spec.get("resource_envelope", {}).get("enabled"):
        trans_events = []
        if run_dir and (run_dir / "events.jsonl").is_file():
            for line in (run_dir / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event.get("phase") == "resource_transition":
                    trans_events.append(event)
        first_high = next((e for e in trans_events if int(e.get("experience", -1)) == 3), None)
        trans_ok = bool(first_high and first_high.get("status") == "ok")
        if summary.get("failure_phase") == "resource_transition":
            trans_ok = False
        if summary.get("unannounced_transition") and int(summary.get("failure_experience") or -1) == 3:
            trans_ok = False
    return {
        "role": role,
        "config": rel,
        "config_id": Path(rel).stem,
        "run_id": summary.get("run_id"),
        "status": summary.get("status") or "not_run",
        "feedback_source": spec["controller"]["feedback_source"],
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
        "supply_wait_ratio": supply,
        "reserved_bytes_by_experience": (spec.get("resource_envelope") or {}).get(
            "reserved_bytes_by_experience"
        ),
        "transition_first_ok": trans_ok,
        "failure_phase": summary.get("failure_phase"),
        "n_experiences_trained": summary.get("n_experiences_trained") or summary.get("n_experiences_run"),
        "n_experiences_evaluated": summary.get("n_experiences_evaluated"),
    }


def run_rels(rels: list[str]) -> None:
    state = (
        json.loads(PROGRESS.read_text())
        if PROGRESS.exists()
        else {"study_id": "pressure_v2", "consumed_s": 0, "active": None, "results": []}
    )
    reconcile_active(state)
    save(PROGRESS, state)
    keep = {"completed", "cuda_oom", "host_oom", "budget_exceeded"}
    todo_paths = []
    todo_specs = []
    for rel in rels:
        previous = next(
            (
                item
                for item in reversed(state.get("results", []))
                if item.get("config") == rel and item.get("status") in keep
            ),
            None,
        )
        if previous:
            print(json.dumps({"config": rel, "status": "dev_artifact_kept", "run_id": previous.get("run_id")}), flush=True)
            continue
        todo_paths.append(ROOT / rel)
        todo_specs.append(load_yaml(ROOT / rel))
    if todo_paths:
        run_config_list(todo_paths, todo_specs, state, PROGRESS, frozen=None)


def write_probe(rows: list[dict[str, Any]]) -> Path:
    payload = {"study_id": "pressure_v2", "schema_version": 1, "runs": rows}
    path = ROOT / "experiments/pressure_v2/probe_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def source_identity() -> dict[str, Any]:
    snap = snapshot_source_tree(ROOT)
    return {
        "dataset_hash": sha256_file(ROOT / "data/manifests/core50_nc_run0.json"),
        "development_dataset_hash": sha256_file(
            ROOT / "data/manifests/core50_nc_run0_dev_split_seed17.json"
        ),
        "source_hash": snap["aggregate_sha256"],
        "design_hash": sha256_file(ROOT / "experiments/pressure_v2/design.yaml"),
    }


def stage_d0() -> list[dict[str, Any]]:
    rows = []
    eval_rels = emit_d0_eval()
    run_rels(eval_rels)
    rows.extend(collect_row(rel, "d0_eval_batch") for rel in eval_rels)
    eval_batch = 32
    eval_ok = [r for r in rows if r["eval_batch"] == 32 and r["status"] == "completed"]
    if not eval_ok:
        eval_batch = 8
    profile = []
    for i in range(2):
        profile.append(
            emit_static_dev(
                role="l_cal",
                quota_mib=256,
                batch=16,
                replay=200,
                eval_batch=eval_batch,
                name=f"lcal_b16_r200_q256_rep{i}",
                seed=0,
            )
        )
    profile.append(
        emit_static_dev(
            role="d0_static_b64",
            quota_mib=256,
            batch=64,
            replay=200,
            eval_batch=eval_batch,
            name="profile_b64_r200_q256",
        )
    )
    profile.append(
        emit_static_dev(
            role="d0_advanced",
            quota_mib=256,
            batch=16,
            replay=200,
            eval_batch=eval_batch,
            plugins="gem_ewc",
            start_enabled=True,
            name="profile_advanced_q256",
        )
    )
    emit_matrix("d0_profile", profile)
    run_rels(profile)
    rows.append(collect_row(profile[0], "l_cal"))
    rows.append(collect_row(profile[1], "l_cal"))
    rows.append(collect_row(profile[2], "d0_static_b64"))
    rows.append(collect_row(profile[3], "d0_advanced"))
    lcal = [r for r in rows if r["role"] == "l_cal" and r["train_reserved_peak_ratio"] is not None]
    peak_ratio = max((r["train_reserved_peak_ratio"] for r in lcal), default=None)
    reserved_peak_mib = None
    if peak_ratio is not None:
        reserved_peak_mib = peak_ratio * 256
        low = max(64, int((reserved_peak_mib / 0.95) // 16 * 16))
        high = int((reserved_peak_mib / 0.80) // 16 * 16) + 16
        candidates = sorted(set([128, 144, 160, low, high, low + 16]))
        candidates = [c for c in candidates if 64 <= c <= 512]
    else:
        candidates = [128, 144, 160, 192, 256]
    scan = []
    for quota in candidates:
        scan.append(
            emit_static_dev(
                role="quota_scan",
                quota_mib=quota,
                batch=16,
                replay=200,
                eval_batch=eval_batch,
                name=f"quota_b16_r200_q{quota}",
            )
        )
    emit_matrix("d0_quota", scan)
    run_rels(scan)
    rows.extend(collect_row(rel, "quota_scan") for rel in scan)
    write_probe(rows)
    return rows


def stage_d1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_completed = {r["eval_batch"] for r in rows if r["role"] == "d0_eval_batch" and r["status"] == "completed"}
    eval_batch = 32 if 32 in eval_completed else (8 if 8 in eval_completed else 32)
    tight_hits = [
        r
        for r in rows
        if r["role"] == "quota_scan"
        and r["status"] == "completed"
        and r.get("train_reserved_peak_ratio") is not None
        and 0.80 <= float(r["train_reserved_peak_ratio"]) <= 0.95
    ]
    if not tight_hits:
        write_probe(rows)
        return rows
    q_tight = min(int(r["quota_mib"]) for r in tight_hits)
    q_loose = 2 * q_tight
    learning = []
    for row in rows:
        if row.get("role") == "l_cal" and row.get("learning_s_by_experience"):
            learning.extend(float(v) for v in row["learning_s_by_experience"])
    l_cal = sorted(learning)[len(learning) // 2] if learning else 30.0
    loose_rel = emit_static_dev(
        role="q_loose_verify",
        quota_mib=q_loose,
        batch=16,
        replay=200,
        eval_batch=eval_batch,
        name=f"loose_verify_q{q_loose}",
    )
    run_rels([loose_rel])
    rows.append(collect_row(loose_rel, "q_loose_verify"))
    if q_loose != 256:
        extra_lcal = []
        for i in range(2):
            extra_lcal.append(
                emit_static_dev(
                    role="l_cal",
                    quota_mib=q_loose,
                    batch=16,
                    replay=200,
                    eval_batch=eval_batch,
                    name=f"lcal_b16_r200_q{q_loose}_rep{i}",
                    seed=0,
                )
            )
        run_rels(extra_lcal)
        rows[:] = [r for r in rows if r.get("role") != "l_cal"]
        rows.extend(collect_row(rel, "l_cal") for rel in extra_lcal)
        learning = []
        for row in rows:
            if row.get("role") == "l_cal" and row.get("learning_s_by_experience"):
                learning.extend(float(v) for v in row["learning_s_by_experience"])
        l_cal = sorted(learning)[len(learning) // 2] if learning else l_cal
    # If 256 already ran as l_cal and q_loose is 256, still keep dedicated verify.
    search_tight = []
    search_loose = []
    for batch, replay in STATIC_GRID:
        search_tight.append(
            emit_static_dev(
                role="static_search_tight",
                quota_mib=q_tight,
                batch=batch,
                replay=replay,
                eval_batch=eval_batch,
                name=f"search_tight_b{batch}_r{replay}_q{q_tight}",
            )
        )
        search_loose.append(
            emit_static_dev(
                role="static_search_loose",
                quota_mib=q_loose,
                batch=batch,
                replay=replay,
                eval_batch=eval_batch,
                name=f"search_loose_b{batch}_r{replay}_q{q_loose}",
            )
        )
    emit_matrix("d1_search", search_tight + search_loose)
    run_rels(search_tight + search_loose)
    rows.extend(collect_row(rel, "static_search_tight") for rel in search_tight)
    rows.extend(collect_row(rel, "static_search_loose") for rel in search_loose)

    controls = []
    for name, latency, m_max in (
        ("O00", 30.0, 4096.0),
        ("O10", float(l_cal), 4096.0),
        ("O01", 30.0, float(q_tight)),
        ("O11", float(l_cal), float(q_tight)),
    ):
        extra = {
            "controller": {
                "enabled": True,
                "plugin_policy": "adaptive",
                "thresholds": {"p": 0.5, "s": 0.5, "latency_s": 30.0, "m_max_mib": m_max},
            },
            "algorithm": {"optional_plugins": "gem_ewc", "optional_start_enabled": False, **PLUGIN_DEFAULTS},
        }
        rel = emit_static_dev(
            role="control_2x2",
            quota_mib=q_tight,
            batch=16,
            replay=200,
            eval_batch=eval_batch,
            plugins="gem_ewc",
            controller=True,
            name=f"control_{name}_q{q_tight}",
            extra=extra,
        )
        spec = load_yaml(ROOT / rel)
        spec["controller"]["thresholds"]["latency_s"] = float(latency)
        spec["controller"]["thresholds"]["m_max_mib"] = m_max
        _write_yaml(ROOT / rel, spec)
        controls.append(rel)
    emit_matrix("d1_control", controls)
    run_rels(controls)
    rows.extend(collect_row(rel, "control_2x2") for rel in controls)

    # DYN reservation: keep transition-first experience completable.
    lcal = next((r for r in rows if r["role"] == "l_cal" and r.get("train_reserved_peak_ratio")), None)
    reserved_peak = (lcal["train_reserved_peak_ratio"] * 256) if lcal else None
    headroom = None
    if reserved_peak is not None:
        headroom = q_tight * 1024**2 - reserved_peak * 1024**2
    high = 0
    if headroom is not None and headroom >= 8 * 1024**2:
        high = int(headroom * 0.5) // (1024**2) * 1024**2
        high = max(high, 8 * 1024**2)
        # leave ~8MiB slack for eval
        high = min(high, int(headroom - 8 * 1024**2))
        high = max(high, 0)
    schedule = default_dyn_schedule(9, high)
    dyn_rel = emit_static_dev(
        role="dyn_reservation_probe",
        quota_mib=q_tight,
        batch=16,
        replay=200,
        eval_batch=eval_batch,
        name=f"dyn_probe_q{q_tight}",
        extra={"resource_envelope": {"enabled": True, "reserved_bytes_by_experience": schedule}},
    )
    run_rels([dyn_rel])
    rows.append(collect_row(dyn_rel, "dyn_reservation_probe"))
    dyn_row = rows[-1]
    if dyn_row.get("status") == "completed":
        dyn_search = []
        for batch, replay in STATIC_GRID:
            dyn_search.append(
                emit_static_dev(
                    role="static_search_dyn",
                    quota_mib=q_tight,
                    batch=batch,
                    replay=replay,
                    eval_batch=eval_batch,
                    name=f"search_dyn_b{batch}_r{replay}_q{q_tight}",
                    extra={"resource_envelope": {"enabled": True, "reserved_bytes_by_experience": schedule}},
                )
            )
        emit_matrix("d1_dyn_search", dyn_search)
        run_rels(dyn_search)
        rows.extend(collect_row(rel, "static_search_dyn") for rel in dyn_search)

    io_extra = {"data_supply": {"record_hashes": True, "profile": "natural_ondemand"}}
    io_off = emit_static_dev(
        role="io_off",
        quota_mib=q_loose,
        batch=16,
        replay=200,
        eval_batch=eval_batch,
        name="io_off_q_loose",
        extra={**io_extra, "prefetch": {"enabled": False, "queue_depth": 2, "pin_memory": False, "num_workers": 0}},
    )
    io_on = emit_static_dev(
        role="io_on",
        quota_mib=q_loose,
        batch=16,
        replay=200,
        eval_batch=eval_batch,
        name="io_on_q_loose",
        extra={**io_extra, "prefetch": {"enabled": True, "queue_depth": 2, "pin_memory": False, "num_workers": 0}},
    )
    emit_matrix("d1_io", [io_off, io_on])
    run_rels([io_off, io_on])
    rows.append(collect_row(io_off, "io_off"))
    rows.append(collect_row(io_on, "io_on"))
    write_probe(rows)
    return rows


def stage_freeze(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        payload = json.loads((ROOT / "experiments/pressure_v2/probe_manifest.json").read_text())
    else:
        payload = {"study_id": "pressure_v2", "runs": rows}
    design = load_yaml(ROOT / "experiments/pressure_v2/design.yaml")
    validate_design(design)
    calibration = calibrate(payload)
    frozen = freeze(calibration, source_identity())
    path = ROOT / "experiments/pressure_v2/frozen_protocol.json"
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    emit(frozen, ROOT)
    design["status"] = "frozen_executable"
    design["executable"] = True
    (ROOT / "experiments/pressure_v2/design.yaml").write_text(
        yaml.safe_dump(design, sort_keys=False), encoding="utf-8"
    )
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["d0", "d1", "freeze", "all"])
    args = parser.parse_args(argv)
    try:
        locks = acquire_locks([ROOT / "runs/.orion_project.lock", ROOT / "runs/.light24.lock"])
    except BlockingIOError:
        parser.error("Another Orion training executor is active")
    try:
        rows: list[dict[str, Any]] = []
        probe = ROOT / "experiments/pressure_v2/probe_manifest.json"
        if probe.exists() and args.stage != "d0":
            rows = json.loads(probe.read_text()).get("runs", [])
        if args.stage in {"d0", "all"}:
            rows = stage_d0()
        if args.stage in {"d1", "all"}:
            rows = stage_d1(rows)
        if args.stage in {"freeze", "all"}:
            try:
                frozen = stage_freeze(rows if rows else None)
                print(json.dumps({"frozen_hash": frozen["frozen_hash"], "coverage": frozen["scenario_coverage"]}, indent=2))
            except Exception as exc:
                print(json.dumps({"freeze_failed": str(exc)}))
                raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    main()
