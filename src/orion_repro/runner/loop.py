"""Single-run training loop: one new-stream pass per experience (A03)."""

from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from orion_repro.benchmarks.dev_split import resolve_feedback_source
from orion_repro.benchmarks.factory import build_benchmark, experience_class_map
from orion_repro.control.controller import ControlState, step_controller
from orion_repro.runner.checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint
from orion_repro.control.urge import UrgeConfig, bytes_to_mib, resolve_controller_coefficients
from orion_repro.evaluation.evaluator import evaluate_domains
from orion_repro.evaluation.domains import eval_protocol, make_eval_domains
from orion_repro.evaluation.metrics import summarize_matrix
from orion_repro.memory.budget import (
    formula_stop_if_infeasible,
    guarded_clip,
    require_cost_model_for_enforcement,
)
from orion_repro.memory.cost_model import MemoryCostModel
from orion_repro.memory.phase_recorder import PhaseRecorder
from orion_repro.memory.probe import ResourceSampler, reset_gpu_peak, snapshot, synchronize_gpu
from orion_repro.memory.resource_envelope import ResourceEnvelope
from orion_repro.provenance import fill_provenance, snapshot_source_tree
from orion_repro.rng import seed_streams
from orion_repro.runner.artifacts import RunArtifacts
from orion_repro.runner.failures import failure_metadata, parse_cuda_oom_request_bytes
from orion_repro.runner.spec import RunSpec, canonical_hash, validate_mapping
from orion_repro.strategies.builder import (
    apply_runtime_config,
    build_model,
    build_optimizer,
    build_strategy,
    plugin_audit,
    replay_occupancy,
    snapshot_plugin_activity,
)
from orion_repro.strategies.capacity import collect_auxiliary_visits
from orion_repro.strategies.toggles import TogglePlugin

ROOT = Path(__file__).resolve().parents[3]


class BudgetExceeded(RuntimeError):
    """Preflight rejected the next-experience configuration."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_torch(spec: dict[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(spec["training"].get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(spec["training"].get("allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(spec["training"].get("cudnn_benchmark", False))
    deterministic = bool(spec["training"].get("deterministic_algorithms", False))
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.set_float32_matmul_precision("highest")


def set_seeds(seed: int) -> None:
    """Deprecated wrapper: only the model seed. Prefer seed_streams(spec)."""
    from orion_repro.rng import seed_python_and_numpy, seed_torch_global

    seed_python_and_numpy(seed)
    seed_torch_global(seed)


def _device(spec: dict[str, Any]) -> torch.device:
    name = spec["training"].get("device", "cuda")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _generate_run_id(spec: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{stamp}_{spec['phase']}_{spec['method_id']}_{spec['dataset']['name']}_{short}"


def _append_registry(row: dict[str, Any]) -> None:
    path = ROOT / "experiments" / "registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _load_cost_model(spec: dict[str, Any]) -> MemoryCostModel | None:
    raw = spec["budget"].get("cost_model_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MemoryCostModel(
        intercept_bytes=float(payload["intercept_bytes"]),
        m_batch_bytes=float(payload["m_batch_bytes"]),
        m_frame_bytes=float(payload["m_frame_bytes"]),
        plugin_gem_ewc_bytes=float(payload.get("plugin_gem_ewc_bytes", 0.0)),
        r2=float(payload.get("r2", 0.0)),
        notes=str(payload.get("notes", "")),
        m_frame_raw_uint8_bytes=float(payload.get("m_frame_raw_uint8_bytes", 3 * 32 * 32)),
        m_frame_latent_f32_bytes=float(payload.get("m_frame_latent_f32_bytes", payload["m_frame_bytes"])),
        resource_for_slope=str(payload.get("resource_for_slope", "device")),
        complete=bool(payload.get("complete", False)),
        error_device_mape=payload.get("error_device_mape"),
        error_host_frame_mae_bytes=payload.get("error_host_frame_mae_bytes"),
        n_device_pairs=int(payload.get("n_device_pairs", 0)),
        n_host_pairs=int(payload.get("n_host_pairs", 0)),
    )


def _toggle_states(strategy) -> dict[str, bool]:
    states: dict[str, bool] = {}
    for plugin in getattr(strategy, "plugins", []):
        if isinstance(plugin, TogglePlugin):
            states[plugin.name] = bool(plugin.enabled)
    return states


def _prefetch_stats(strategy) -> dict[str, float | int]:
    plugin = getattr(strategy, "_orion_prefetch_plugin", None)
    if plugin is None:
        return {
            "prefetch_wait_s": 0.0,
            "prefetch_dropped_stale": 0,
            "prefetch_batches": 0,
            "prefetch_plan_mode": "unused",
            "prefetch_planned_n": 0,
        }
    return {
        "prefetch_wait_s": float(getattr(plugin, "last_wait_s", 0.0)),
        "prefetch_dropped_stale": int(getattr(plugin, "last_dropped_stale", 0)),
        "prefetch_batches": int(getattr(plugin, "last_batches", 0)),
        "prefetch_plan_mode": str(getattr(plugin, "last_plan_mode", "")),
        "prefetch_planned_n": int(getattr(plugin, "last_planned_n", 0)),
        "prefetch_produce_s": float(getattr(plugin, "last_produce_s", 0.0)),
    }


def _reservation_schedule(spec: dict[str, Any], n_run: int) -> list[int]:
    env = spec.get("resource_envelope") or {}
    if not env or not env.get("enabled"):
        return [0] * n_run
    sched = env.get("reserved_bytes_by_experience")
    if sched is None:
        return [0] * n_run
    values = [int(x) for x in sched]
    if len(values) != n_run:
        raise ValueError(
            f"reserved_bytes_by_experience length {len(values)} != n_run {n_run}"
        )
    return values


def _consumption_digest(strategy) -> dict[str, Any]:
    plugin = getattr(strategy, "_orion_prefetch_plugin", None)
    if plugin is None:
        return {}
    digest = dict(getattr(plugin, "last_digest", {}) or {})
    digest.setdefault("produce_s", float(getattr(plugin, "last_produce_s", 0.0)))
    digest.setdefault("wait_s", float(getattr(plugin, "last_wait_s", 0.0)))
    return digest


def _current_replay_batch(spec: dict[str, Any], ctrl_state: ControlState | None) -> int:
    if ctrl_state is None:
        return int(spec["training"]["replay_batch"])
    return int(ctrl_state.new_batch)


def _admit_config(
    spec: dict[str, Any],
    *,
    new_batch: int,
    replay_capacity: int,
    replay_batch: int,
    advanced: bool,
    cost_model: MemoryCostModel | None,
    experience_index: int,
) -> dict[str, Any]:
    """Shared admission for experience 0, later experiences, and static baselines."""
    enforcement = spec["budget"]["enforcement"]
    limit_bytes = spec["budget"].get("limit_bytes")
    min_batch = int(spec["controller"].get("min_batch", 1))
    max_batch = spec["controller"].get("max_batch")
    resource = spec["budget"]["controlled_resource"]
    representation = str(spec["replay"].get("representation", "avalanche_buffer"))
    predicted_bytes = None
    if cost_model is not None:
        predicted_bytes = cost_model.predict_bytes(
            new_batch,
            replay_capacity,
            replay_batch=replay_batch,
            advanced=advanced,
            resource=resource,
            representation=representation,
        )
    if enforcement in {"observed_only", "device_allocator_enforced"}:
        return {
            "accepted": True,
            "reason": enforcement,
            "predicted_bytes": predicted_bytes,
            "limit_bytes": limit_bytes,
            "enforcement": enforcement,
            "experience_index": experience_index,
            "budget_scientifically_valid": False,
        }
    require_cost_model_for_enforcement(enforcement, cost_model)
    if enforcement == "formula_stop":
        guard = formula_stop_if_infeasible(
            suggested_new_batch=new_batch,
            suggested_replay=replay_capacity,
            predicted_bytes=predicted_bytes,
            limit_bytes=limit_bytes,
            min_batch=min_batch,
            max_batch=max_batch,
        )
    elif enforcement == "guarded_clip":
        if predicted_bytes is None or limit_bytes is None or cost_model is None:
            raise BudgetExceeded("cost_model_required")
        guard = guarded_clip(
            suggested_new_batch=new_batch,
            suggested_replay=replay_capacity,
            predicted_bytes=predicted_bytes,
            limit_bytes=int(limit_bytes),
            min_batch=min_batch,
            bytes_per_batch_step=max(1, int(round(cost_model.m_batch_bytes))),
            bytes_per_replay_item=max(1, int(round(cost_model.m_frame_raw_uint8_bytes))),
            max_batch=max_batch,
        )
        if guard.accepted and guard.reason == "clipped":
            return {
                "accepted": True,
                "reason": guard.reason,
                "predicted_bytes": predicted_bytes,
                "limit_bytes": limit_bytes,
                "enforcement": enforcement,
                "experience_index": experience_index,
                "applied_new_batch": guard.applied_new_batch,
                "applied_replay": guard.applied_replay,
                "budget_scientifically_valid": bool(getattr(cost_model, "complete", False)),
            }
    else:
        raise BudgetExceeded(f"unsupported enforcement {enforcement}")
    if not guard.accepted:
        raise BudgetExceeded(guard.reason)
    return {
        "accepted": True,
        "reason": guard.reason,
        "predicted_bytes": predicted_bytes,
        "limit_bytes": limit_bytes,
        "enforcement": enforcement,
        "experience_index": experience_index,
        "budget_scientifically_valid": bool(cost_model is not None and cost_model.complete),
    }


def run_from_spec(spec: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    os_environ_set = __import__("os").environ
    os_environ_set.setdefault("WANDB_MODE", "disabled")
    os_environ_set.setdefault("WANDB_DISABLED", "true")
    validate_mapping(spec, require_provenance=False)
    if not spec.get("run_id"):
        spec["run_id"] = _generate_run_id(spec)
    run_id = spec["run_id"]
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    arts = RunArtifacts(run_dir)
    from orion_repro.runner.reuse import reuse_identity
    run_reuse_identity = reuse_identity(spec, ROOT)
    status = "running"
    summary: dict[str, Any] = {"run_id": run_id, "status": status}
    sampler: ResourceSampler | None = None
    log_handles: list = []
    t_run0 = time.monotonic()
    failure_phase = "setup"
    trained_experiences = 0
    recorder: PhaseRecorder | None = None
    envelope: ResourceEnvelope | None = None
    current_unannounced = False
    quota_bytes = spec.get("budget", {}).get("limit_bytes")
    try:
        configure_torch(spec)
        device = _device(spec)
        from orion_repro.memory.enforcement import install_device_quota  # noqa: F401
        envelope = ResourceEnvelope(device)
        quota = envelope.install_quota(spec["budget"], device)
        quota_bytes = spec["budget"].get("limit_bytes")
        arts.write_json("budget_enforcement.json", {"quota": quota})
        recorder = PhaseRecorder(
            arts,
            quota_bytes=quota_bytes,
            reservation_getter=lambda: envelope.reservation_bytes if envelope is not None else 0,
            device=device,
        )
        recorder.begin("setup", None)
        code_snapshot = snapshot_source_tree(ROOT)
        spec = fill_provenance(spec, root=ROOT, snapshot=code_snapshot)
        feedback_source = resolve_feedback_source(spec)
        validate_mapping(spec, require_provenance=spec.get("phase") == "formal")
        streams = seed_streams(spec)
        model_seed = streams["model"]

        arts.write_json("code_snapshot.json", {k: v for k, v in code_snapshot.items() if k != "files"})
        arts.write_json("code_snapshot_files.json", {"files": code_snapshot["files"]})
        arts.write_json("resolved_config.yaml.json", spec)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
        )
        arts.write_json(
            "environment.json",
            {
                "executable": sys.executable,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "seeds": streams,
                "feedback_source": feedback_source,
                "code_revision_or_snapshot": spec.get("code_revision_or_snapshot"),
                "environment_lock_sha256": spec.get("environment_lock_sha256"),
                "dataset_manifest_sha256": spec.get("dataset_manifest_sha256"),
            },
        )

        warmup_bs = min(8, int(spec["training"]["new_batch"]))
        if device.type == "cuda":
            dummy = torch.randn(warmup_bs, 3, 32, 32, device=device)
            conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
            with torch.no_grad():
                _ = conv(dummy)
            synchronize_gpu(device)
            del dummy, conv
            set_seeds(model_seed)

        benchmark = build_benchmark(spec)
        if getattr(benchmark, "orion_split_records", None):
            split_payload = {"records": benchmark.orion_split_records}
            arts.write_json("split_manifest.json", split_payload)
            from orion_repro.provenance import sha256_file
            spec["dataset_manifest_sha256"] = sha256_file(run_dir / "split_manifest.json")
            environment_record = json.loads((run_dir / "environment.json").read_text())
            environment_record["dataset_manifest_sha256"] = spec["dataset_manifest_sha256"]
            arts.write_json("environment.json", environment_record)
            arts.write_json("resolved_config.yaml.json", spec)
            (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        class_map = experience_class_map(benchmark)
        protocol = eval_protocol(benchmark, class_map)
        arts.write_json(
            "manifest_refs.json",
            {
                "dataset": spec["dataset"],
                "class_map": class_map,
                "n_train_experiences": len(benchmark.train_stream),
                "n_test_experiences": len(benchmark.test_stream),
                "eval_protocol": protocol,
                "feedback_source": feedback_source,
                "config_path": str(config_path) if config_path else None,
                "spec_hash": canonical_hash(spec),
                "dataset_manifest_sha256": spec.get("dataset_manifest_sha256"),
                "code_revision_or_snapshot": spec.get("code_revision_or_snapshot"),
            },
        )

        model = build_model(spec).to(device)
        optimizer = build_optimizer(model, spec)
        avalanche_log = (run_dir / "avalanche.log").open("w", encoding="utf-8")
        log_handles.append(avalanche_log)
        strategy = build_strategy(
            model, optimizer, spec, device=device, log_file=avalanche_log
        )
        cost_model = _load_cost_model(spec)
        arts.write_json(
            "plugins.json",
            {
                "order": plugin_audit(strategy),
                "cost_model": cost_model.as_dict() if cost_model is not None else None,
            },
        )

        n_exp = len(benchmark.train_stream)
        limit = int(spec["dataset"]["experience_limit"])
        n_run = min(n_exp, limit)
        matrix = np.full((n_run, n_run), np.nan, dtype=np.float64)
        totals = np.zeros((n_run, n_run), dtype=np.float64)
        correct_mat = np.zeros((n_run, n_run), dtype=np.float64)

        ctrl_cfg = None
        ctrl_state = None
        initial_mode = (
            "advanced"
            if bool(spec["algorithm"].get("optional_start_enabled", False))
            else "default"
        )
        if spec["controller"]["enabled"]:
            c = spec["controller"]
            coef = resolve_controller_coefficients(c)
            c["coefficients"] = coef
            ctrl_cfg = UrgeConfig(
                kp=float(coef["kp"]),
                ks=float(coef["ks"]),
                kl=float(coef["kl"]),
                km=float(coef["km"]),
                p_th=float(c["thresholds"]["p"]),
                s_th=float(c["thresholds"]["s"]),
                latency_th_s=float(c["thresholds"]["latency_s"]),
                m_max_mib=float(c["thresholds"]["m_max_mib"]),
                thr0=float(c["thr0"]),
                delta=float(c["delta"]),
                alpha=float(c["updates"]["alpha"]),
                beta=float(c["updates"]["beta"]),
                equal_uses_gt=bool(c.get("equal_uses_gt", True)),
            )
            ctrl_state = ControlState(
                mb=float(spec["controller"]["mb0"]),
                mr=float(spec["controller"]["mr0"]),
                new_batch=int(spec["training"]["new_batch"]),
                replay_capacity=int(spec["replay"]["capacity"]),
                optimizer_mode=initial_mode,
            )

        interval = float(spec["measurement"]["sample_interval_ms"]) / 1000.0
        sampler = ResourceSampler(interval_s=max(interval, 0.05), device=device)
        sampler.start()
        if recorder is not None:
            recorder.end("setup", None)
        _append_registry({"run_id": run_id, "status": "running", "run_dir": str(run_dir)})

        last_metrics = None
        completed_experiences = 0
        t_online0 = time.monotonic()
        setup_s = t_online0 - t_run0
        start_k = 0
        resume_path = spec["measurement"].get("resume_checkpoint")
        if resume_path:
            ckpt_file = Path(resume_path)
            if not ckpt_file.is_absolute():
                ckpt_file = ROOT / ckpt_file
            payload = load_checkpoint(ckpt_file)
            apply_checkpoint(strategy, payload, device=device)
            start_k = int(payload["completed_experiences"])
            matrix = np.asarray(payload["matrix"], dtype=np.float64)
            correct_mat = np.asarray(payload["correct_mat"], dtype=np.float64)
            totals = np.asarray(payload["totals"], dtype=np.float64)
            if payload.get("controller") and spec["controller"]["enabled"]:
                cstate = payload["controller"]
                ctrl_state = ControlState(
                    mb=float(cstate["mb"]),
                    mr=float(cstate["mr"]),
                    new_batch=int(cstate["new_batch"]),
                    replay_capacity=int(cstate["replay_capacity"]),
                    optimizer_mode=str(cstate["optimizer_mode"]),
                )
                apply_runtime_config(
                    strategy,
                    new_batch=ctrl_state.new_batch,
                    replay_capacity=ctrl_state.replay_capacity,
                    replay_batch=ctrl_state.new_batch,
                    optimizer_mode=ctrl_state.optimizer_mode,
                )
            completed_experiences = start_k
            arts.event(
                {
                    "utc": _utc_now(),
                    "monotonic_s": time.monotonic(),
                    "phase": "resume_checkpoint",
                    "path": str(ckpt_file),
                    "completed_experiences": start_k,
                }
            )
        reservation_schedule = _reservation_schedule(spec, n_run)
        for k in range(start_k, n_run):
            prev_reservation = reservation_schedule[k - 1] if k > 0 else 0
            current_unannounced = int(reservation_schedule[k]) != int(prev_reservation)
            failure_phase = "resource_transition"
            if recorder is not None and envelope is not None:
                recorder.begin("resource_transition", k)
                trans = envelope.transition(k, reservation_schedule[k])
                rec_transition = recorder.end("resource_transition", k)
                arts.event(
                    {
                        "utc": _utc_now(),
                        "monotonic_s": time.monotonic(),
                        "experience": k,
                        "phase": "resource_transition",
                        "unannounced_transition": current_unannounced,
                        "duration_s": rec_transition.duration_s,
                        **trans.as_dict(),
                    }
                )
                if trans.status != "ok":
                    raise torch.cuda.OutOfMemoryError(
                        f"Tried to allocate {reservation_schedule[k] / (1024 ** 2):.2f} MiB for resource envelope"
                    )
            cur_batch = int(strategy.train_mb_size)
            cur_replay = int(
                spec["replay"]["capacity"] if ctrl_state is None else ctrl_state.replay_capacity
            )
            cur_replay_batch = _current_replay_batch(spec, ctrl_state)
            cur_advanced = (
                (ctrl_state.optimizer_mode == "advanced") if ctrl_state else (initial_mode == "advanced")
            )
            admission = _admit_config(
                spec,
                new_batch=cur_batch,
                replay_capacity=cur_replay,
                replay_batch=cur_replay_batch if k > 0 else 0,
                advanced=cur_advanced,
                cost_model=cost_model,
                experience_index=k,
            )
            arts.event(
                {
                    "utc": _utc_now(),
                    "monotonic_s": time.monotonic(),
                    "experience": k,
                    "phase": "admission",
                    **admission,
                    **replay_occupancy(strategy),
                }
            )
            if admission.get("applied_new_batch") is not None:
                apply_runtime_config(
                    strategy,
                    new_batch=int(admission["applied_new_batch"]),
                    replay_capacity=int(admission.get("applied_replay", cur_replay)),
                    replay_batch=int(admission["applied_new_batch"]),
                    optimizer_mode=ctrl_state.optimizer_mode if ctrl_state else initial_mode,
                )
            exp = benchmark.train_stream[k]
            sampler.set_phase("learning", k)
            if recorder is not None:
                recorder.begin("training", k)
            else:
                reset_gpu_peak(device)
            snap0 = snapshot("learning_start", k, device=device)
            arts.append_csv(arts._resource, snap0.as_row())
            arts.event(
                {
                    "utc": _utc_now(),
                    "monotonic_s": time.monotonic(),
                    "experience": k,
                    "phase": "train_start",
                    "new_batch": int(strategy.train_mb_size),
                    "replay_capacity": int(
                        spec["replay"]["capacity"] if ctrl_state is None else ctrl_state.replay_capacity
                    ),
                    "optimizer_mode": ctrl_state.optimizer_mode if ctrl_state else initial_mode,
                    "plugins": _toggle_states(strategy),
                    "external_reservation_bytes": envelope.reservation_bytes if envelope is not None else 0,
                }
            )
            t0 = time.monotonic()
            failure_phase = "training"
            strategy.train(exp, num_workers=int(spec["prefetch"].get("num_workers", 0)))
            synchronize_gpu(device)
            learning_s = time.monotonic() - t0
            trained_experiences = k + 1
            train_phase = recorder.end("training", k) if recorder is not None else None
            learning_peak_rss = sampler.phase_peak_rss
            learning_peak_gpu = sampler.phase_peak_gpu_alloc
            reserved_peak_gpu = sampler.phase_peak_gpu_reserved
            if train_phase is not None:
                if train_phase.allocated_peak_bytes:
                    learning_peak_gpu = int(train_phase.allocated_peak_bytes)
                if train_phase.reserved_peak_bytes:
                    reserved_peak_gpu = int(train_phase.reserved_peak_bytes)
            pf = _prefetch_stats(strategy)
            snap1 = snapshot("learning_end", k, device=device)
            arts.append_csv(arts._resource, snap1.as_row())
            arts.append_jsonl(
                arts._plugin_activity,
                {"experience": k, "phase": "training", **snapshot_plugin_activity(strategy)},
            )
            arts.append_jsonl(
                arts._consumption,
                {"experience": k, **_consumption_digest(strategy)},
            )

            sampler.set_phase("evaluation", k)
            if recorder is not None:
                recorder.begin("evaluation", k)
            else:
                reset_gpu_peak(device)
            t1 = time.monotonic()
            protocol = eval_protocol(benchmark, class_map)
            domains = make_eval_domains(benchmark, class_map, k)
            failure_phase = "evaluation"
            rows = evaluate_domains(
                strategy.model,
                domains,
                device=device,
                batch_size=int(spec["training"]["eval_batch"]),
                num_workers=0,
            )
            synchronize_gpu(device)
            evaluation_s = time.monotonic() - t1
            if recorder is not None:
                recorder.end("evaluation", k)
            if protocol == "shared_test_temporal":
                acc_row = rows[0]
                for i in range(k + 1):
                    matrix[k, i] = acc_row.accuracy
                    totals[k, i] = acc_row.total
                    correct_mat[k, i] = acc_row.correct
                    arts.append_csv(
                        arts._matrix,
                        {
                            "train_experience": k,
                            "eval_domain": i,
                            "correct": acc_row.correct,
                            "total": acc_row.total,
                            "accuracy": acc_row.accuracy,
                            "protocol_id": spec["protocol_id"],
                            "eval_protocol": protocol,
                        },
                    )
            else:
                for row in rows:
                    matrix[k, row.eval_domain] = row.accuracy
                    totals[k, row.eval_domain] = row.total
                    correct_mat[k, row.eval_domain] = row.correct
                    arts.append_csv(
                        arts._matrix,
                        {
                            "train_experience": k,
                            "eval_domain": row.eval_domain,
                            "correct": row.correct,
                            "total": row.total,
                            "accuracy": row.accuracy,
                            "protocol_id": spec["protocol_id"],
                            "eval_protocol": protocol,
                        },
                    )

            metrics = summarize_matrix(matrix[: k + 1, : k + 1], totals[k, : k + 1])
            last_metrics = metrics
            completed_experiences = k + 1
            shared_test_accuracy = (
                float(rows[0].accuracy) if protocol == "shared_test_temporal" else None
            )
            mem_bytes = snap1.gpu_alloc_peak_bytes or snap1.gpu_alloc_bytes or snap1.proc_rss_bytes
            resource_name = spec["budget"]["controlled_resource"]
            if resource_name == "host":
                mem_bytes = max(
                    int(learning_peak_rss or 0),
                    int(snap1.proc_rss_bytes + snap1.children_rss_bytes),
                )
            elif resource_name == "device":
                mem_bytes = max(
                    int(learning_peak_gpu or 0),
                    int(snap1.gpu_alloc_peak_bytes or snap1.gpu_alloc_bytes or 0),
                )
            reservation_bytes = envelope.reservation_bytes if envelope is not None else 0
            memory_mib = bytes_to_mib(mem_bytes)
            memory_mib_model = bytes_to_mib(max(0, int(mem_bytes) - int(reservation_bytes)))
            occ = replay_occupancy(strategy)
            visits = dict(strategy._orion_prefetch_plugin.last_visits)
            visits.update(collect_auxiliary_visits(strategy))
            if spec["algorithm"]["base"] == "er" and spec["algorithm"].get("optional_plugins") == "none":
                visits["auxiliary_visits"] = 0
                visits["auxiliary_visit_scope"] = "no_auxiliary_forward_plain_er"
            if spec["algorithm"]["base"] == "lr" and spec["algorithm"].get("optional_plugins") == "none":
                from orion_repro.strategies.latent_replay import LatentReplayPlugin

                latent = next(p for p in strategy.plugins if isinstance(p, LatentReplayPlugin))
                visits.update(
                    replay_visits=latent.replay_visits,
                    auxiliary_visits=latent.auxiliary_visits,
                    auxiliary_visit_scope="latent_cache_feature_forward",
                )

            arts.append_csv(
                arts._metrics,
                {
                    "experience_index": k,
                    "p_diag": metrics.p_diag,
                    "s_initial": metrics.s_initial,
                    "avg_seen_accuracy": metrics.avg_seen_accuracy,
                    "forgetting_max": metrics.forgetting_max,
                    "stability_max": metrics.stability_max,
                    "current_experience_accuracy": metrics.current_experience_accuracy,
                    "eval_protocol": protocol,
                    "shared_test_accuracy": shared_test_accuracy,
                    "learning_s": learning_s,
                    "evaluation_s": evaluation_s,
                    "memory_mib": memory_mib,
                    "memory_mib_model": memory_mib_model,
                    "allocated_peak_bytes": int(mem_bytes) if mem_bytes is not None else None,
                    "reserved_peak_bytes": int(reserved_peak_gpu or 0) or None,
                    "external_reservation_bytes": int(reservation_bytes),
                    "prefetch_produce_s": pf.get("prefetch_produce_s", 0.0),
                    "new_batch": int(strategy.train_mb_size),
                    "replay_capacity": int(
                        ctrl_state.replay_capacity if ctrl_state else spec["replay"]["capacity"]
                    ),
                    "optimizer_mode": ctrl_state.optimizer_mode if ctrl_state else initial_mode,
                    "prefetch_wait_s": pf["prefetch_wait_s"],
                    "prefetch_dropped_stale": pf["prefetch_dropped_stale"],
                    "prefetch_batches": pf["prefetch_batches"],
                    "prefetch_plan_mode": pf.get("prefetch_plan_mode"),
                    "prefetch_planned_n": pf.get("prefetch_planned_n"),
                    "replay_occupancy": occ.get("replay_occupancy"),
                    "replay_max_size": occ.get("replay_max_size"),
                    "latent_occupancy": occ.get("latent_occupancy"),
                    "plugin_state_bytes": json.dumps(occ.get("plugin_state_bytes") or {}, default=str),
                    **visits,

                },
            )

            if ctrl_cfg is not None and ctrl_state is not None:
                t_c0 = time.monotonic()
                failure_phase = "controller"
                if recorder is not None:
                    recorder.begin("controller", k)
                mb_before = ctrl_state.mb
                mr_before = ctrl_state.mr
                decision = step_controller(
                    ctrl_cfg,
                    ctrl_state,
                    t=k,
                    plasticity=metrics.p_diag,
                    stability=metrics.s_initial,
                    latency_s=learning_s,
                    memory_mib=memory_mib,
                    m_batch=float(spec["controller"]["m_batch"]),
                    m_frame=float(spec["controller"]["m_frame"]),
                )
                if k + 1 == n_run:
                    if recorder is not None:
                        recorder.end("controller", k)
                    arts.control({"t": k, "urge": decision.urge, "thr": decision.thr,
                        "factors": decision.factors, "plasticity": metrics.p_diag,
                        "stability": metrics.s_initial, "latency_s": learning_s,
                        "memory_mib": memory_mib, "memory_mib_model": memory_mib_model,
                        "mb_before": mb_before, "mr_before": mr_before,
                        "mb_next": decision.mb_next, "mr_next": decision.mr_next,
                        "suggested_new_batch": decision.suggested_new_batch,
                        "suggested_replay": decision.suggested_replay,
                        "suggested_optimizer_mode": decision.suggested_optimizer_mode,
                        "applied": None,
                        "reason": "last_experience", "controller_s": time.monotonic() - t_c0,
                        "reconfigure_s": 0.0})
                    continue
                from orion_repro.control.ablation import applied_optimizer_mode
                effective_mode = applied_optimizer_mode(spec["controller"], decision.suggested_optimizer_mode)
                predicted_bytes = None
                if cost_model is not None:
                    predicted_bytes = cost_model.predict_bytes(
                        decision.suggested_new_batch,
                        decision.suggested_replay,
                        replay_batch=max(0, decision.suggested_new_batch),
                        advanced=effective_mode == "advanced",
                        resource=spec["budget"]["controlled_resource"],
                        representation=str(spec["replay"].get("representation", "avalanche_buffer")),
                    )
                applied_batch = decision.suggested_new_batch
                applied_replay = decision.suggested_replay
                applied_mode = effective_mode
                guard_reason = "ok"
                enforcement = spec["budget"]["enforcement"]
                limit_bytes = spec["budget"].get("limit_bytes")
                min_batch = int(spec["controller"].get("min_batch", 1))
                max_batch = spec["controller"].get("max_batch")
                if enforcement == "formula_stop":
                    guard = formula_stop_if_infeasible(
                        suggested_new_batch=decision.suggested_new_batch,
                        suggested_replay=decision.suggested_replay,
                        predicted_bytes=predicted_bytes,
                        limit_bytes=limit_bytes,
                        min_batch=min_batch,
                        max_batch=max_batch,
                    )
                    guard_reason = guard.reason
                    if not guard.accepted:
                        arts.control(
                            {
                                "t": k,
                                "urge": decision.urge,
                                "thr": decision.thr,
                                "factors": decision.factors,
                                "suggested_new_batch": decision.suggested_new_batch,
                                "suggested_replay": decision.suggested_replay,
                                "predicted_bytes": predicted_bytes,
                                "limit_bytes": limit_bytes,
                                "guard_reason": guard.reason,
                                "enforcement": enforcement,
                                "cost_model_complete": bool(
                                    cost_model.complete if cost_model is not None else False
                                ),
                            }
                        )
                        raise BudgetExceeded(guard.reason)
                elif enforcement == "guarded_clip":
                    if predicted_bytes is None or limit_bytes is None or cost_model is None:
                        raise BudgetExceeded("cost_model_required")
                    guard = guarded_clip(
                        suggested_new_batch=decision.suggested_new_batch,
                        suggested_replay=decision.suggested_replay,
                        predicted_bytes=predicted_bytes,
                        limit_bytes=int(limit_bytes),
                        min_batch=min_batch,
                        bytes_per_batch_step=max(1, int(round(cost_model.m_batch_bytes))),
                        bytes_per_replay_item=max(1, int(round(cost_model.m_frame_raw_uint8_bytes))),
                        max_batch=max_batch,
                    )
                    guard_reason = guard.reason
                    if not guard.accepted:
                        arts.control(
                            {
                                "t": k,
                                "guard_reason": guard.reason,
                                "enforcement": enforcement,
                                "predicted_bytes": predicted_bytes,
                                "limit_bytes": limit_bytes,
                            }
                        )
                        raise BudgetExceeded(guard.reason)
                    applied_batch = guard.applied_new_batch
                    applied_replay = guard.applied_replay
                elif enforcement not in {"observed_only", "device_allocator_enforced"}:
                    raise BudgetExceeded(f"unsupported enforcement {enforcement}")
                ctrl_state = ControlState(
                    mb=decision.mb_next,
                    mr=decision.mr_next,
                    new_batch=applied_batch,
                    replay_capacity=applied_replay,
                    optimizer_mode=applied_mode,
                )
                controller_s = time.monotonic() - t_c0
                if recorder is not None:
                    recorder.end("controller", k)
                    recorder.begin("reconfiguration", k)
                t_reconfigure0 = time.monotonic()
                failure_phase = "reconfiguration"
                occ_after = apply_runtime_config(
                    strategy,
                    new_batch=ctrl_state.new_batch,
                    replay_capacity=ctrl_state.replay_capacity,
                    replay_batch=ctrl_state.new_batch,
                    optimizer_mode=ctrl_state.optimizer_mode,
                )
                synchronize_gpu(device)
                reconfigure_s = time.monotonic() - t_reconfigure0
                if recorder is not None:
                    recorder.end("reconfiguration", k)
                arts.control(
                    {
                        "t": k,
                        "urge": decision.urge,
                        "thr": decision.thr,
                        "factors": decision.factors,
                        "mb_before": mb_before,
                        "mr_before": mr_before,
                        "mb_next": decision.mb_next,
                        "mr_next": decision.mr_next,
                        "suggested_new_batch": decision.suggested_new_batch,
                        "suggested_replay": decision.suggested_replay,
                        "suggested_optimizer_mode": decision.suggested_optimizer_mode,
                        "plugin_policy": spec["controller"].get("plugin_policy", "adaptive"),
                        "applied_new_batch": ctrl_state.new_batch,
                        "applied_replay": ctrl_state.replay_capacity,
                        "applied_optimizer_mode": ctrl_state.optimizer_mode,
                        "applied_optional_plugins": occ_after.get("applied_optional_plugins", {}),
                        "replay_requested": occ_after.get("replay_requested"),
                        "replay_occupancy": occ_after.get("replay_occupancy"),
                        "replay_max_size": occ_after.get("replay_max_size"),
                        "predicted_bytes": predicted_bytes,
                        "limit_bytes": limit_bytes,
                        "guard_reason": guard_reason,
                        "enforcement": enforcement,
                        "controller_s": controller_s,
                        "reconfigure_s": reconfigure_s,
                        "plasticity": metrics.p_diag,
                        "stability": metrics.s_initial,
                        "latency_s": learning_s,
                        "memory_mib": memory_mib,
                        "memory_mib_model": memory_mib_model,
                        "plugins": _toggle_states(strategy),
                    }
                )
            else:
                arts.control({"t": k, "controller": "skipped", "reason": "disabled"})
            if spec["measurement"].get("checkpoint_policy") == "experience_boundary":
                ckpt_dir = run_dir / "checkpoints"
                save_checkpoint(
                    ckpt_dir / f"experience_{k}.pt",
                    strategy=strategy,
                    ctrl_state=ctrl_state,
                    completed_experiences=k + 1,
                    matrix=matrix,
                    correct_mat=correct_mat,
                    totals=totals,
                    extra={"run_id": run_id},
                )
                save_checkpoint(
                    ckpt_dir / "latest.pt",
                    strategy=strategy,
                    ctrl_state=ctrl_state,
                    completed_experiences=k + 1,
                    matrix=matrix,
                    correct_mat=correct_mat,
                    totals=totals,
                    extra={"run_id": run_id},
                )

        status = "completed"
        summary = {
            "run_id": run_id,
            "status": status,
            "phase": spec["phase"],
            "method_id": spec["method_id"],
            "dataset": spec["dataset"]["name"],
            "n_experiences_run": completed_experiences,
            "eval_protocol": locals().get("protocol"),
            "p_diag": last_metrics.p_diag if last_metrics else None,
            "s_initial": last_metrics.s_initial if last_metrics else None,
            "shared_test_accuracy": locals().get("shared_test_accuracy"),
            "online_total_s": time.monotonic() - locals().get("t_online0", time.monotonic()),
            "setup_s": locals().get("setup_s", time.monotonic() - t_run0),
            "run_total_s": time.monotonic() - t_run0,
            "timing_schema": "online_loop_v2",
            "resumed": bool(locals().get("resume_path")),
            "run_dir": str(run_dir),
        }
        return summary
    except torch.cuda.OutOfMemoryError as exc:
        status = "cuda_oom"
        request = parse_cuda_oom_request_bytes(str(exc))
        last = recorder.last if recorder is not None else None
        summary = {"run_id": run_id, "status": status, "traceback": traceback.format_exc(),
                   **failure_metadata(
                       spec,
                       phase=failure_phase,
                       experience=locals().get("k"),
                       trained=trained_experiences,
                       evaluated=locals().get("completed_experiences", 0),
                       request_bytes=request,
                       quota_bytes=quota_bytes,
                       unannounced_transition=bool(current_unannounced and failure_phase in {
                           "training", "evaluation", "resource_transition"
                       }),
                       allocated_peak_bytes=None if last is None else last.allocated_peak_bytes,
                       reserved_peak_bytes=None if last is None else last.reserved_peak_bytes,
                       external_reservation_bytes=None if envelope is None else envelope.reservation_bytes,
                   )}
        return summary
    except BudgetExceeded as exc:
        status = "budget_exceeded"
        summary = {
            "run_id": run_id,
            "status": status,
            "reason": str(exc),
            "n_experiences_run": locals().get("completed_experiences", 0),
            "p_diag": locals().get("last_metrics").p_diag if locals().get("last_metrics") else None,
            "s_initial": locals().get("last_metrics").s_initial if locals().get("last_metrics") else None,
            "online_total_s": time.monotonic() - locals().get("t_online0", time.monotonic()),
            "setup_s": locals().get("setup_s", time.monotonic() - t_run0),
            "run_total_s": time.monotonic() - t_run0,
            "timing_schema": "online_loop_v2",
            "run_dir": str(run_dir),
        }
        return summary
    except Exception:
        status = "implementation_error"
        summary = {"run_id": run_id, "status": status, "traceback": traceback.format_exc()}
        raise
    finally:
        for handle in log_handles:
            try:
                handle.close()
            except Exception:
                pass
        if envelope is not None:
            envelope.close()
        if sampler is not None:
            sampler.stop()
            for row in sampler.rows:
                arts.append_csv(arts._resource, row.as_row())
        summary["status"] = status
        summary["reuse_identity"] = run_reuse_identity
        arts.write_json("summary.json", summary)
        arts.close()
        _append_registry(summary)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    spec = RunSpec.from_file(path)
    data = spec.resolved_copy()
    if not data.get("run_id"):
        data["run_id"] = _generate_run_id(data)
    summary = run_from_spec(data, config_path=path)
    print(json.dumps(summary, indent=2, default=str))
    if summary.get("status") not in {"completed", "cuda_oom", "host_oom", "budget_exceeded"}:
        if summary.get("status") != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
