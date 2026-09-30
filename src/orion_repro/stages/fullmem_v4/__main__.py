"""CLI for fullmem_v4 G1/G2. Help is read-only regarding v3 queues."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from orion_repro.stages.fullmem_v4.boot import parse_extlinux, proposed_mem_label, proposed_mem64, read_live_extlinux, rollback_to_primary
from orion_repro.stages.fullmem_v4.constants import EXTLINUX_PATH, ORION_PYTHON, STUDY_ID, repo_root
from orion_repro.stages.fullmem_v4.executor import executor_loop, pause, resume
from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.probes import run_shared_pool_probe
from orion_repro.stages.fullmem_v4.queue import ensure_dirs, list_inbox, load_state, stage_paths, submit
from orion_repro.stages.fullmem_v4.util import StageError, assert_write_path, atomic_write_json, atomic_write_text, utc_now


def cmd_identity(args) -> int:
    payload = collect_identity(include_cuda=not args.no_cuda)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.save:
        path = assert_write_path(Path(args.save) if Path(args.save).is_absolute() else repo_root() / args.save)
        atomic_write_json(path, payload)
    return 0


def cmd_boot_plan(args) -> int:
    original = read_live_extlinux()
    parsed = parse_extlinux(original)
    mem = str(args.mem)
    label = str(args.label or f"orion-mem{mem.lower()}")
    menu = str(args.menu or f"Orion fullmem_v4 mem={mem}")
    proposed = proposed_mem_label(original, mem=mem, label=label, menu=menu)
    rollback = rollback_to_primary(proposed)
    paths = ensure_dirs()
    stamp = utc_now().replace(":", "")
    dest = paths["reports"] / "boot" / stamp
    dest = assert_write_path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_text(dest / "extlinux.conf.original", original)
    atomic_write_text(dest / "extlinux.conf.proposed", proposed)
    atomic_write_text(dest / "extlinux.conf.rollback", rollback)
    report = {
        "study_id": STUDY_ID,
        "created_at_utc": utc_now(),
        "live_path": str(EXTLINUX_PATH),
        "current_default": parsed.get("default"),
        "labels": sorted((parsed.get("labels") or {}).keys()),
        "proposed_default": parse_extlinux(proposed).get("default"),
        "proposed_label": label,
        "mem": mem,
        "nvpmodel_untouched": True,
        "apply": False,
        "notes": [
            "This command does not write /boot.",
            "Apply only after review: copy proposed over /boot/extlinux/extlinux.conf with sudo, keep original LABEL primary.",
            "Do not change nvpmodel.",
            "Keep existing mem= LABELs; only DEFAULT points at the new test entry.",
        ],
    }
    atomic_write_json(dest / "plan.json", report)
    print(json.dumps({"dir": str(dest), **report}, indent=2))
    return 0


def cmd_submit(args) -> int:
    raw = json.loads(Path(args.task).read_text(encoding="utf-8"))
    result = submit(raw)
    print(json.dumps(result, indent=2))
    return 0


def cmd_status(args) -> int:
    paths = ensure_dirs()
    state = load_state()
    inbox = [{"task_id": t["task_id"], "seq": t.get("seq"), "kind": t.get("kind")} for t in list_inbox()]
    payload = {
        "state": state,
        "inbox": inbox,
        "running": sorted(p.name for p in paths["running"].glob("*.json")),
        "done": sorted(p.name for p in paths["done"].glob("*.json")),
        "identity": collect_identity(include_cuda=False),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_pause(args) -> int:
    print(json.dumps(pause(args.reason), indent=2))
    return 0


def cmd_resume(args) -> int:
    print(json.dumps(resume(), indent=2))
    return 0


def cmd_executor(args) -> int:
    return executor_loop(once=args.once)


def cmd_dummy(args) -> int:
    import time

    time.sleep(max(0.0, float(args.seconds)))
    marker = Path(args.marker)
    if not marker.is_absolute():
        marker = repo_root() / marker
    marker = assert_write_path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"ok {utc_now()}\n", encoding="utf-8")
    print(json.dumps({"marker": str(marker), "ok": True}))
    return 0


def cmd_probe(args) -> int:
    result = run_shared_pool_probe(max_gib=float(args.max_gib), chunk_mib=int(args.chunk_mib))
    print(json.dumps({k: result[k] for k in ("probe", "steps", "output", "mixed") if k in result}, indent=2, default=str))
    return 0 if all(v == "ok" for v in result.get("steps", {}).values()) else 2


def cmd_install_service(args) -> int:
    src = Path(__file__).resolve().parent / "systemd" / "orion-fullmem-v4-executor.service"
    dest_dir = Path.home() / ".config" / "systemd" / "user"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copyfile(src, dest)
    print(json.dumps({"copied": str(dest), "src": str(src), "interpreter": ORION_PYTHON}))
    return 0


def cmd_g2_inventory(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_inventory import save_inventory

    print(json.dumps(save_inventory(), indent=2, default=str))
    return 0


def cmd_g2_physical(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_physical import run_physical_audit

    payload = run_physical_audit()
    print(json.dumps({"output": payload.get("output"), "board": payload.get("board"), "warning": payload.get("formula_warning", {}).get("note")}, indent=2))
    return 0


def cmd_g2_prepare_cifar10(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_inventory import prepare_cifar10_without_manifest_rewrite

    print(json.dumps(prepare_cifar10_without_manifest_rewrite(), indent=2, default=str))
    return 0


def cmd_g2_prepare_endless(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_inventory import prepare_endless_without_manifest_rewrite

    print(json.dumps(prepare_endless_without_manifest_rewrite(args.scenario), indent=2, default=str))
    return 0


def cmd_g2_occupancy(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_occupancy import occupancy_report

    payload = occupancy_report()
    print(json.dumps({"output": payload.get("output"), "n_members": len(payload.get("members") or []), "pressure_scene": payload.get("pressure_scene")}, indent=2))
    return 0


def cmd_h4_calibrate(args) -> int:
    from orion_repro.stages.fullmem_v4.h4_calibrate import calibrate_h4_hold

    payload = calibrate_h4_hold()
    print(
        json.dumps(
            {
                "output": payload.get("output"),
                "high_bytes": payload.get("high_bytes"),
                "identifiable": payload.get("identifiable"),
                "capacity_id": payload.get("capacity_id"),
                "recovered_after_release": payload.get("recovered_after_release"),
            },
            indent=2,
        )
    )
    return 0


def cmd_g2_prepare_core50_splits(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_inventory import prepare_core50_dev_splits_without_manifest_rewrite

    print(json.dumps(prepare_core50_dev_splits_without_manifest_rewrite(), indent=2, default=str))
    return 0


def cmd_g2_lcal(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_lcal import lcal_report

    payload = lcal_report()
    print(json.dumps({"output": payload.get("output"), "n_members": len(payload.get("members") or [])}, indent=2))
    return 0


def cmd_g2_idle(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_idle import idle_baseline

    payload = idle_baseline()
    print(json.dumps({"output": payload.get("output"), "windows": len(payload.get("windows") or [])}, indent=2))
    return 0


def cmd_g2_cost(args) -> int:
    from orion_repro.stages.fullmem_v4.g2_cost import cost_estimate

    payload = cost_estimate()
    print(json.dumps({"output": payload.get("output"), "design_slots": payload.get("design_slots_before_dedup")}, indent=2))
    return 0


def cmd_dummy_fail(args) -> int:
    print(json.dumps({"ok": False, "reason": args.reason}))
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m orion_repro.stages.fullmem_v4")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ident = sub.add_parser("identity")
    ident.add_argument("--no-cuda", action="store_true")
    ident.add_argument("--save")
    ident.set_defaults(func=cmd_identity)
    boot = sub.add_parser("boot-plan")
    boot.add_argument("--mem", default="64G")
    boot.add_argument("--label", default=None)
    boot.add_argument("--menu", default=None)
    boot.set_defaults(func=cmd_boot_plan)
    submit_p = sub.add_parser("submit")
    submit_p.add_argument("--task", required=True)
    submit_p.set_defaults(func=cmd_submit)
    sub.add_parser("status").set_defaults(func=cmd_status)
    pause_p = sub.add_parser("pause")
    pause_p.add_argument("--reason", default="manual")
    pause_p.set_defaults(func=cmd_pause)
    sub.add_parser("resume").set_defaults(func=cmd_resume)
    exe = sub.add_parser("executor-run")
    exe.add_argument("--once", action="store_true")
    exe.set_defaults(func=cmd_executor)
    dummy = sub.add_parser("dummy")
    dummy.add_argument("--seconds", default="1")
    dummy.add_argument("--marker", required=True)
    dummy.set_defaults(func=cmd_dummy)
    probe = sub.add_parser("probe-shared-pool")
    probe.add_argument("--max-gib", default="8")
    probe.add_argument("--chunk-mib", default="256")
    probe.set_defaults(func=cmd_probe)
    sub.add_parser("install-user-service").set_defaults(func=cmd_install_service)
    sub.add_parser("g2-inventory").set_defaults(func=cmd_g2_inventory)
    sub.add_parser("g2-physical").set_defaults(func=cmd_g2_physical)
    sub.add_parser("g2-prepare-cifar10").set_defaults(func=cmd_g2_prepare_cifar10)
    endless = sub.add_parser("g2-prepare-endless")
    endless.add_argument("--scenario", default="all")
    endless.set_defaults(func=cmd_g2_prepare_endless)
    sub.add_parser("g2-occupancy").set_defaults(func=cmd_g2_occupancy)
    sub.add_parser("h4-calibrate").set_defaults(func=cmd_h4_calibrate)
    sub.add_parser("g2-prepare-core50-splits").set_defaults(func=cmd_g2_prepare_core50_splits)
    sub.add_parser("g2-lcal").set_defaults(func=cmd_g2_lcal)
    sub.add_parser("g2-idle-baseline").set_defaults(func=cmd_g2_idle)
    sub.add_parser("g2-cost").set_defaults(func=cmd_g2_cost)
    fail = sub.add_parser("dummy-fail")
    fail.add_argument("--reason", default="g1_implementation_error_injection")
    fail.set_defaults(func=cmd_dummy_fail)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except StageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
