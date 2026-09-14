from __future__ import annotations

import argparse
import json
from pathlib import Path

from orion_repro.runner.spec import RunSpec, SpecError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    try:
        spec = RunSpec.from_file(path)
    except SpecError as exc:
        raise SystemExit(f"INVALID SPEC: {exc}") from exc
    print(json.dumps({"ok": True, "path": str(path), "phase": spec.phase, "method_id": spec.method_id}, indent=2))


if __name__ == "__main__":
    main()
