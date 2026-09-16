"""Matrix-driven reporting. Never glob-and-overwrite by method/seed."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from orion_repro.stages.effectiveness_v3.constants import STUDY_ID
from orion_repro.stages.effectiveness_v3.context import StageContext
from orion_repro.stages.effectiveness_v3.coverage import audit_scenarios
from orion_repro.stages.effectiveness_v3.util import StageError, atomic_write_json, atomic_write_text


PAIR_CORE = (
    "dataset",
    "split_manifest",
    "model_name",
    "optimizer",
    "seed",
    "eval_batch",
    "precision",
    "quota_bytes",
    "source_hash",
    "frozen_hash",
)


def collect(matrix: dict[str, Any], context: StageContext) -> dict[str, Any]:
    context.reject_foreign_study(matrix)
    progress = {"results": []}
    if context.progress_path.exists():
        progress = json.loads(context.progress_path.read_text(encoding="utf-8"))
    attempts = list(progress.get("results") or [])
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        cell = row.get("cell_id")
        if cell:
            by_cell[cell].append(row)
    joined = []
    for cell in matrix.get("cells") or []:
        cell_id = cell.get("cell_id")
        rows = by_cell.get(cell_id, [])
        joined.append({"cell": cell, "attempts": rows, "n_attempts": len(rows)})
    return {
        "study_id": STUDY_ID,
        "revision": context.revision,
        "frozen_hash": matrix.get("frozen_hash"),
        "design_slots": matrix.get("design_slots"),
        "eligible_n": matrix.get("eligible_n"),
        "excluded_n": matrix.get("excluded_n"),
        "cells": joined,
        "attempts": attempts,
    }


def validate_pairs(left: dict[str, Any], right: dict[str, Any], *, allow: set[str] | None = None) -> None:
    allow = allow or set()
    for key in PAIR_CORE:
        if key in allow:
            continue
        if left.get(key) != right.get(key):
            raise StageError(f"pairing mismatch on {key}: {left.get(key)!r} vs {right.get(key)!r}")
    if left.get("frozen_hash") and right.get("frozen_hash") and left["frozen_hash"] != right["frozen_hash"]:
        raise StageError("refusing to pair across frozen protocols")


def render(bundle: dict[str, Any], context: StageContext, *, calibration: dict[str, Any] | None = None) -> Path:
    context.ensure_revision_dirs()
    coverage = audit_scenarios(calibration or {"study_id": STUDY_ID, "scenario_coverage": {}})
    attempts_path = context.report_dir / "attempts.csv"
    context.assert_output_path(attempts_path)
    rows = []
    for item in bundle.get("cells") or []:
        cell = item["cell"]
        if not item["attempts"]:
            rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "group": cell.get("group"),
                    "dataset": cell.get("dataset"),
                    "method": cell.get("method"),
                    "seed": cell.get("seed"),
                    "variant": cell.get("variant"),
                    "status": cell.get("exclude_reason") or "not_run",
                    "run_id": "",
                    "attempt_n": 0,
                }
            )
            continue
        for i, attempt in enumerate(item["attempts"]):
            rows.append(
                {
                    "cell_id": cell.get("cell_id"),
                    "group": cell.get("group"),
                    "dataset": cell.get("dataset"),
                    "method": cell.get("method"),
                    "seed": cell.get("seed"),
                    "variant": cell.get("variant"),
                    "status": attempt.get("status"),
                    "run_id": attempt.get("run_id") or "",
                    "attempt_n": i + 1,
                    "elapsed_s": attempt.get("elapsed_s"),
                }
            )
    if rows:
        with attempts_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(context.report_dir / "scenario_coverage.json", coverage)
    lines = [
        f"# effectiveness_v3 {context.revision}",
        "",
        f"design_slots={bundle.get('design_slots')} eligible={bundle.get('eligible_n')} excluded={bundle.get('excluded_n')}",
        "",
        "判定尚未用正式结果填写。失败与未建立场景保留。",
        "",
    ]
    results = context.report_dir / "RESULTS.md"
    context.assert_output_path(results)
    atomic_write_text(results, "\n".join(lines) + "\n")
    return results
