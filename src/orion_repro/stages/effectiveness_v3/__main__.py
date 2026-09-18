"""CLI for effectiveness_v3. Help is read-only and never launches old formal matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orion_repro.stages.effectiveness_v3.calibration import calibrate
from orion_repro.stages.effectiveness_v3.constants import DEFAULT_REVISION, STUDY_ID
from orion_repro.stages.effectiveness_v3.context import StageContext, inspect_payload, require_orion_interpreter
from orion_repro.stages.effectiveness_v3.development import (
    empty_manifest,
    emit_probe_configs,
    ingest_batch,
    plan_probes,
    record_host_capability,
)
from orion_repro.stages.effectiveness_v3.executor import dry_run_matrix, execute_matrix
from orion_repro.stages.effectiveness_v3.matrix import emit as emit_matrix
from orion_repro.stages.effectiveness_v3.protocol import freeze, load_or_build_identity, write_frozen
from orion_repro.stages.effectiveness_v3.report import collect, render
from orion_repro.stages.effectiveness_v3.schema import load_design
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json, load_yaml


ROOT = Path(__file__).resolve().parents[4]


def _context(args) -> StageContext:
    return StageContext(root=ROOT, revision=getattr(args, "revision", DEFAULT_REVISION))


def cmd_inspect(args) -> int:
    payload = inspect_payload(_context(args))
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["historical_protection"]["ok"] else 1


def cmd_develop(args) -> int:
    context = _context(args)
    design = load_design(context.design_path)
    manifest_path = context.revision_dir / "probe_manifest.json"
    evidence = empty_manifest(context.revision)
    if manifest_path.exists():
        evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
        context.reject_foreign_study(evidence)
    evidence = record_host_capability(evidence)
    if not args.execute:
        context.ensure_revision_dirs()
        batch = plan_probes(design, evidence, context)
        emit_probe_configs(batch, context)
        context.assert_output_path(manifest_path)
        atomic_write_json(manifest_path, evidence)
        print(
            json.dumps(
                {
                    "n_probes": len(batch.probes),
                    "blocked_reason": batch.blocked_reason,
                    "probe_ids": [p.probe_id for p in batch.probes],
                    "execute": False,
                },
                indent=2,
            )
        )
        return 0
    require_orion_interpreter()
    context.ensure_revision_dirs()
    waves = 0
    last_ids: list[str] = []
    while waves < int(args.max_waves):
        batch = plan_probes(design, evidence, context)
        if not batch.probes:
            context.assert_output_path(manifest_path)
            atomic_write_json(manifest_path, evidence)
            print(
                json.dumps(
                    {
                        "done": batch.blocked_reason is None,
                        "blocked_reason": batch.blocked_reason,
                        "n_runs": len(evidence.get("runs") or []),
                        "waves": waves,
                    },
                    indent=2,
                )
            )
            return 2 if batch.blocked_reason else 0
        emit_probe_configs(batch, context)
        matrix = load_yaml(context.revision_dir / "dev_batch.yaml")
        result = execute_matrix(matrix, context, frozen=None)
        evidence = ingest_batch(batch, context, evidence)
        context.assert_output_path(manifest_path)
        atomic_write_json(manifest_path, evidence)
        last_ids = [p.probe_id for p in batch.probes]
        recent = [r for r in evidence["runs"] if r.get("probe_id") in set(last_ids)]
        if any(r.get("status") == "implementation_error" for r in recent):
            print(json.dumps({"paused": True, "reason": "implementation_error", "probe_ids": last_ids, **result}, indent=2))
            return 2
        waves += 1
        print(json.dumps({"wave": waves, "probe_ids": last_ids, **result}, indent=2), flush=True)
    context.assert_output_path(manifest_path)
    atomic_write_json(manifest_path, evidence)
    print(json.dumps({"paused": True, "reason": "max_waves", "waves": waves}, indent=2))
    return 0


def cmd_freeze(args) -> int:
    context = _context(args)
    from orion_repro.stages.effectiveness_v3.readiness import require_freeze_review
    require_freeze_review(context)
    design = load_design(context.design_path)
    manifest_path = context.revision_dir / "probe_manifest.json"
    if not manifest_path.exists():
        raise StageError("no probe_manifest.json; development calibration is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = calibrate(manifest)
    context.ensure_revision_dirs()
    cal_path = context.revision_dir / "calibration.json"
    context.assert_output_path(cal_path)
    atomic_write_json(cal_path, calibration)
    identity = load_or_build_identity(context)
    frozen = freeze(design, calibration, identity, revision=context.revision)
    path = write_frozen(frozen, context)
    print(json.dumps({"frozen_hash": frozen["frozen_hash"], "path": str(path.relative_to(context.root))}, indent=2))
    return 0


def cmd_emit(args) -> int:
    context = _context(args)
    frozen_path = context.revision_dir / "frozen_protocol.json"
    if not frozen_path.exists():
        raise StageError("frozen_protocol.json missing; refuse to emit formal configs")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    matrix = emit_matrix(frozen, context)
    print(
        json.dumps(
            {
                "design_slots": matrix["design_slots"],
                "eligible_n": matrix["eligible_n"],
                "excluded_n": matrix["excluded_n"],
            },
            indent=2,
        )
    )
    return 0


def cmd_run(args) -> int:
    context = _context(args)
    if args.matrix:
        matrix_path = Path(args.matrix)
        if not matrix_path.is_absolute():
            matrix_path = context.root / matrix_path
        matrix = load_yaml(matrix_path) if matrix_path.suffix in {".yaml", ".yml"} else json.loads(matrix_path.read_text())
    else:
        matrix_path = context.revision_dir / "matrix.json"
        if not matrix_path.exists():
            raise StageError("no matrix.json; emit after freeze first")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    context.reject_foreign_study(matrix)
    if args.dry_run:
        print(json.dumps(dry_run_matrix(matrix, context), indent=2))
        return 0
    frozen = None
    frozen_path = context.revision_dir / "frozen_protocol.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if matrix.get("name") == "formal_54" or matrix.get("study_id") != STUDY_ID:
        raise StageError("refusing to execute a non-effectiveness_v3 or old formal_54 matrix")
    result = execute_matrix(matrix, context, frozen=frozen)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_report(args) -> int:
    context = _context(args)
    matrix_path = context.revision_dir / "matrix.json"
    if not matrix_path.exists():
        raise StageError("no matrix.json")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    bundle = collect(matrix, context)
    cal = None
    cal_path = context.revision_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    path = render(bundle, context, calibration=cal)
    print(json.dumps({"results": str(path.relative_to(context.root))}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orion_repro.stages.effectiveness_v3",
        description="effectiveness_v3 stage entry. Does not run pressure_v2 or light24 matrices.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--revision", default=DEFAULT_REVISION)
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect = sub.add_parser("inspect", parents=[common], help="read-only design and isolation check; no writes")
    inspect.set_defaults(func=cmd_inspect)
    develop = sub.add_parser("develop", parents=[common], help="plan or execute development probes")
    mode = develop.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    develop.add_argument("--max-waves", type=int, default=40)
    develop.set_defaults(func=cmd_develop)
    freeze_p = sub.add_parser("freeze", parents=[common], help="freeze protocol from development evidence")
    freeze_p.set_defaults(func=cmd_freeze)
    emit = sub.add_parser("emit", parents=[common], help="emit formal matrix after freeze")
    emit.set_defaults(func=cmd_emit)
    run = sub.add_parser("run", parents=[common], help="execute an effectiveness_v3 matrix")
    run.add_argument("--matrix")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report", parents=[common], help="matrix-driven report")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except StageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
