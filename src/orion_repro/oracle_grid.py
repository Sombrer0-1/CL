"""Paper §5.1 Oracle grid: 7 batch × 6 replay = 42 cells (A08, A15)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ORACLE_BATCHES = [16, 32, 64, 128, 256, 512, 1024]
ORACLE_REPLAY = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


def grid_cells() -> list[dict]:
    cells = []
    for batch in ORACLE_BATCHES:
        for replay in ORACLE_REPLAY:
            cells.append(
                {
                    "new_batch": batch,
                    "replay_capacity": replay,
                    "replay_batch": batch,
                    "cell_id": f"b{batch}_r{replay}",
                }
            )
    return cells


def emit_configs(base_path: Path, out_dir: Path) -> list[Path]:
    import copy

    import yaml

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cell in grid_cells():
        spec = copy.deepcopy(base)
        spec["method_id"] = "oracle_reconstructed"
        spec["experiment_id"] = "E03"
        spec["training"]["new_batch"] = int(cell["new_batch"])
        spec["training"]["replay_batch"] = int(cell["replay_batch"])
        spec["replay"]["capacity"] = int(cell["replay_capacity"])
        spec["oracle_cell_id"] = cell["cell_id"]
        dest = out_dir / f"{cell['cell_id']}.yaml"
        dest.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        written.append(dest)
    return written


def main() -> None:
    cells = grid_cells()
    assert len(cells) == 42
    out = ROOT / "configs" / "oracle" / "grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cells": len(cells),
        "batches": ORACLE_BATCHES,
        "replay": ORACLE_REPLAY,
        "selection_rule": "A15: Pareto then max 0.5P+0.5S, then shorter learning_s, then smaller memory, then dict order",
        "note": "Generator only. Do not treat listing as a completed Oracle search.",
        "cells": cells,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_cells": len(cells)}, indent=2))
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-from")
    parser.add_argument("--out-dir", default="configs/oracle/cifar10_er_dev")
    args, _ = parser.parse_known_args()
    if args.emit_from:
        paths = emit_configs(Path(args.emit_from), ROOT / args.out_dir)
        print(json.dumps({"emitted": len(paths), "out_dir": str(ROOT / args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
