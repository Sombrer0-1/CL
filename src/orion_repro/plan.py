"""Plan a run matrix without training (PLAN §12)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from orion_repro.runner.spec import SpecError, load_yaml, validate_mapping

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    matrix = load_yaml(Path(args.matrix))
    configs = [Path(p) if not Path(p).is_absolute() else Path(p) for p in matrix.get("configs", [])]
    configs = [ROOT / p if not p.is_absolute() else p for p in configs]
    rows = []
    for path in configs:
        rec = {"config": str(path), "exists": path.exists()}
        if path.exists():
            try:
                data = load_yaml(path)
                validate_mapping(data, require_provenance=False)
                rec["ok"] = True
                rec["method_id"] = data.get("method_id")
                rec["phase"] = data.get("phase")
                rec["unresolved"] = False
            except SpecError as exc:
                rec["ok"] = False
                rec["error"] = str(exc)
                rec["unresolved"] = True
        else:
            rec["ok"] = False
            rec["error"] = "missing file"
        rows.append(rec)
    report = {
        "matrix": str(args.matrix),
        "n_configs": len(configs),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "dry_run": True,
        "rows": rows,
        "note": "dry-run does not train; registry reuse is not applied until run_matrix",
    }
    print(json.dumps(report, indent=2))
    if args.dry_run and report["n_ok"] != report["n_configs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
