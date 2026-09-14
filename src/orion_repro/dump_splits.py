"""Persist actual class orders / stream sizes for frozen manifests (A17/A14)."""

from __future__ import annotations

import json
from pathlib import Path

from orion_repro.benchmarks.factory import build_benchmark, experience_class_map
from orion_repro.evaluation.domains import class_sets_disjoint, eval_protocol
from orion_repro.runner.spec import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def dump_from_config(config_path: Path, out_path: Path) -> dict:
    spec = load_yaml(config_path)
    benchmark = build_benchmark(spec)
    class_map = experience_class_map(benchmark)
    protocol = eval_protocol(benchmark, class_map)
    train_sizes = [int(len(exp.dataset)) for exp in benchmark.train_stream]
    test_sizes = [int(len(exp.dataset)) for exp in benchmark.test_stream]
    payload = {
        "dataset": spec["dataset"]["name"],
        "n_experiences_in_config": spec["dataset"]["n_experiences"],
        "n_experiences": len(class_map),
        "n_train_experiences": len(benchmark.train_stream),
        "n_test_experiences": len(benchmark.test_stream),
        "eval_protocol": protocol,
        "feedback_source": spec["controller"].get("feedback_source"),
        "split_manifest": spec["dataset"].get("split_manifest"),
        "split_dir": spec["dataset"].get("split_dir"),
        "class_sets_disjoint": class_sets_disjoint(class_map),
        "stream_seed": spec["seeds"]["stream"],
        "shuffle": spec["dataset"].get("shuffle"),
        "scenario": spec["dataset"].get("scenario"),
        "run": spec["dataset"].get("run"),
        "object_level": spec["dataset"].get("object_level"),
        "fixed_class_order_in_config": spec["dataset"].get("fixed_class_order"),
        "class_map": class_map,
        "flat_class_order": [c for exp in class_map for c in exp],
        "n_classes_per_experience": [len(cs) for cs in class_map],
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
        "n_train_total": int(sum(train_sizes)),
        "n_test_total": int(sum(test_sizes)),
        "config": str(config_path),
    }
    notes: list[str] = []
    if spec["controller"].get("feedback_source") == "development_val_seen":
        notes.append(
            "Development split from official train only. "
            f"n_train_total={int(sum(train_sizes))} is not the official 50k train set; "
            "n_test_total is held-out train val, not the official test set."
        )
    if str(spec["dataset"]["name"]).startswith("core50"):
        notes.append(
            "Paper Table 2 image count 164866 equals mini train+test. "
            "Development val uses held-out s*/o* sequences, not official test_filelist. "
            "mini=True selects 32x32 files, not a different sample count."
        )
    if notes:
        payload["note"] = " ".join(notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = dump_from_config(Path(args.config), Path(args.out))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
