"""Emit frozen paper_feedback configs and matrices (PLAN M6 / E03 / E04).

Does not train. Oracle selected cells are omitted until A15 search completes.
LR × GEM/AGEM/GSS is recorded as not_implemented rather than a silent combo.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from orion_repro.oracle_grid import emit_configs, grid_cells

ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_ID = "paper_feedback_diag_initial_v1"
ALIGNMENT = "A01-A22-formal_v1"

OFFICIAL_MANIFEST = {
    "splitcifar10": "data/manifests/paper_feedback/cifar10_official.json",
    "splitcifar100": "data/manifests/paper_feedback/cifar100_official.json",
    "core50_nc": "data/manifests/core50_nc_run0.json",
    "core50_ni": "data/manifests/core50_ni_run0.json",
    "core50_nic": "data/manifests/core50_nic_run0.json",
}

E03_CONTEXTS = {
    "splitcifar10": "configs/development/cifar10_er_static.yaml",
    "splitcifar100": "configs/development/cifar100_er_static.yaml",
    "core50_ni": "configs/development/core50_ni_er_static.yaml",
    "core50_nc": "configs/development/core50_nc_er_static.yaml",
    "core50_nic": "configs/development/core50_nic_er_static.yaml",
}

E04_ALGO_TEMPLATES = {
    "gem": "configs/development/core50_nc_gem_9exp.yaml",
    "agem": "configs/development/core50_nc_agem_9exp.yaml",
    "gss": "configs/development/core50_nc_gss_9exp.yaml",
}

METHOD_TEMPLATES = {
    "max_a_reconstructed": "configs/development/cifar10_max_a_reconstructed.yaml",
    "max_p_reconstructed": "configs/development/cifar10_max_p_reconstructed.yaml",
    "lr_reconstructed": "configs/development/cifar10_lr_reconstructed.yaml",
    "orion_formula": "configs/development/cifar10_orion_formula.yaml",
}

SEEDS = (0, 1, 2)

LR_SKIP_BASES = {"gem", "agem", "gss"}


def _load(path: str | Path) -> dict[str, Any]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _dump(path: Path, spec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _to_formal_common(spec: dict[str, Any], *, seed: int, experiment_id: str, claim_ids: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    out["phase"] = "formal"
    out["protocol_id"] = PROTOCOL_ID
    out["experiment_id"] = experiment_id
    out["claim_ids"] = list(claim_ids)
    out["alignment_version"] = ALIGNMENT
    out["seeds"] = {key: int(seed) for key in ("model", "stream", "replay", "augmentation")}
    out["controller"]["feedback_source"] = "official_test_seen"
    out["prefetch"]["enabled"] = False
    out["prefetch"]["queue_depth"] = 1
    out["prefetch"]["num_workers"] = 0
    ds_name = out["dataset"]["name"]
    out["dataset"]["split_manifest"] = OFFICIAL_MANIFEST[ds_name]
    out["dataset"].pop("split_dir", None)
    out["dataset_manifest_sha256"] = None
    out["code_revision_or_snapshot"] = None
    out["environment_lock_sha256"] = None
    out["run_id"] = None
    return out


def _overlay_dataset(method_spec: dict[str, Any], context_spec: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(method_spec)
    out["dataset"] = copy.deepcopy(context_spec["dataset"])
    out["model"]["num_classes"] = context_spec["model"]["num_classes"]
    return out


def _apply_method_on_base(spec: dict[str, Any], method_id: str, base: str) -> dict[str, Any] | None:
    if method_id == "lr_reconstructed" and base in LR_SKIP_BASES:
        return None
    out = copy.deepcopy(spec)
    out["method_id"] = method_id
    if method_id == "orion_formula":
        out["controller"]["enabled"] = True
        out["training"]["new_batch"] = 16
        out["training"]["replay_batch"] = 16
        out["replay"]["capacity"] = 200
        out["controller"]["mb0"] = 16.0
        out["controller"]["mr0"] = 200.0
        out["controller"]["thresholds"]["latency_s"] = 30.0
        out["controller"]["thresholds"]["m_max_mib"] = 4096.0
        if base in LR_SKIP_BASES:
            out["algorithm"]["optional_plugins"] = "none"
    elif method_id == "max_p_reconstructed":
        out["controller"]["enabled"] = False
        out["training"]["new_batch"] = 256
        out["training"]["replay_batch"] = 256
        out["replay"]["capacity"] = 10
        out["controller"]["mb0"] = 256.0
        out["controller"]["mr0"] = 10.0
        if base == "er":
            out["algorithm"]["base"] = "er"
            out["algorithm"]["optional_plugins"] = "none"
        elif base in LR_SKIP_BASES:
            out["algorithm"]["optional_plugins"] = "none"
    elif method_id == "max_a_reconstructed":
        out["controller"]["enabled"] = False
        out["training"]["new_batch"] = 1
        out["training"]["replay_batch"] = 1
        out["replay"]["capacity"] = 200
        out["controller"]["mb0"] = 1.0
        out["controller"]["mr0"] = 200.0
        if base == "er":
            out["algorithm"]["base"] = "er"
            out["algorithm"]["optional_plugins"] = "gem_ewc"
            out["algorithm"]["optional_start_enabled"] = True
            out["algorithm"]["patterns_per_exp"] = 50
            out["algorithm"]["memory_strength"] = 0.5
            out["algorithm"]["ewc_lambda"] = 100.0
        elif base == "gem":
            out["algorithm"]["optional_plugins"] = "ewc"
            out["algorithm"]["optional_start_enabled"] = True
            out["algorithm"]["ewc_lambda"] = 100.0
        elif base in {"agem", "gss"}:
            out["algorithm"]["optional_plugins"] = "ewc"
            out["algorithm"]["optional_start_enabled"] = True
            out["algorithm"]["ewc_lambda"] = 100.0
    elif method_id == "lr_reconstructed":
        out["controller"]["enabled"] = False
        out["algorithm"]["base"] = "lr"
        out["algorithm"]["optional_plugins"] = "none"
        out["training"]["new_batch"] = 16
        out["training"]["replay_batch"] = 16
        out["replay"]["capacity"] = 200
        out["replay"]["representation"] = "latent_layer2"
    elif method_id == "er_static":
        out["controller"]["enabled"] = False
        out["algorithm"]["base"] = "er"
        out["algorithm"]["optional_plugins"] = "none"
        out["training"]["new_batch"] = 16
        out["training"]["replay_batch"] = 16
        out["replay"]["capacity"] = 200
    else:
        raise ValueError(method_id)
    return out


def emit_online() -> dict[str, Any]:
    written: list[str] = []
    skipped: list[dict[str, Any]] = []
    e03: list[str] = []
    e04: list[str] = []
    e02: list[str] = []

    for context_name, context_path in E03_CONTEXTS.items():
        context = _load(context_path)
        for method_id, method_path in METHOD_TEMPLATES.items():
            method = _load(method_path)
            for seed in SEEDS:
                merged = _overlay_dataset(method, context)
                applied = _apply_method_on_base(merged, method_id, base="er")
                assert applied is not None
                spec = _to_formal_common(
                    applied, seed=seed, experiment_id="E03", claim_ids=["E03", "C03"]
                )
                rel = f"configs/formal/e03/{context_name}/{method_id}/seed{seed}.yaml"
                _dump(ROOT / rel, spec)
                written.append(rel)
                e03.append(rel)

    for base, algo_path in E04_ALGO_TEMPLATES.items():
        algo = _load(algo_path)
        for method_id in METHOD_TEMPLATES:
            for seed in SEEDS:
                applied = _apply_method_on_base(algo, method_id, base=base)
                if applied is None:
                    skipped.append(
                        {
                            "context": f"core50_nc_{base}",
                            "method_id": method_id,
                            "seed": seed,
                            "reason": (
                                "LR × GEM/AGEM/GSS is not a defined reconstruction: "
                                "latent buffer vs raw gradient constraint are not mixed silently"
                            ),
                        }
                    )
                    continue
                spec = _to_formal_common(
                    applied, seed=seed, experiment_id="E04", claim_ids=["E04", "C04"]
                )
                rel = f"configs/formal/e04/core50_nc_{base}/{method_id}/seed{seed}.yaml"
                _dump(ROOT / rel, spec)
                written.append(rel)
                e04.append(rel)

    er = _load(E03_CONTEXTS["splitcifar10"])
    for method_id, method_path in (
        ("er_static", "configs/development/cifar10_er_static.yaml"),
        ("orion_formula", "configs/development/cifar10_orion_formula.yaml"),
    ):
        method = _load(method_path)
        for seed in SEEDS:
            merged = _overlay_dataset(method, er)
            applied = _apply_method_on_base(merged, method_id, base="er")
            spec = _to_formal_common(
                applied, seed=seed, experiment_id="E02", claim_ids=["E02", "C02"]
            )
            rel = f"configs/formal/e02/splitcifar10/{method_id}/seed{seed}.yaml"
            _dump(ROOT / rel, spec)
            written.append(rel)
            e02.append(rel)

    def write_matrix(name: str, description: str, configs: list[str]) -> str:
        rel = f"experiments/{name}.yaml"
        payload = {"name": name, "description": description, "configs": configs}
        _dump(ROOT / rel, payload)
        return rel

    matrices = {
        "e02": write_matrix(
            "formal_e02",
            "Frozen paper_feedback CIFAR10 ER static vs Orion, 3 seeds. Prefetch off.",
            e02,
        ),
        "e03": write_matrix(
            "formal_e03_online",
            "Frozen paper_feedback E03: 5 datasets × 4 reconstructed online methods × 3 seeds.",
            e03,
        ),
        "e04": write_matrix(
            "formal_e04_online",
            "Frozen paper_feedback E04 GEM/AGEM/GSS × MAX-A/MAX-P/Orion × 3 seeds. LR cells skipped.",
            e04,
        ),
    }
    skip_path = ROOT / "experiments" / "formal_skipped.json"
    skip_path.write_text(json.dumps({"n": len(skipped), "cells": skipped}, indent=2), encoding="utf-8")
    return {
        "n_written": len(written),
        "n_e02": len(e02),
        "n_e03": len(e03),
        "n_e04": len(e04),
        "n_skipped": len(skipped),
        "matrices": matrices,
        "skipped_path": str(skip_path.relative_to(ROOT)),
    }


def emit_oracle_search() -> dict[str, Any]:
    """Emit 8×42 development-identity Oracle search configs (A15, seed 0).

    Search stays on development_val_seen. Selected cells are copied into
    paper_feedback after freeze+search, not here.
    """
    written = []
    matrices = {}
    for context_name, context_path in E03_CONTEXTS.items():
        out_dir = ROOT / "configs" / "oracle" / f"{context_name}_er_dev"
        paths = emit_configs(ROOT / context_path, out_dir)
        rels = [str(p.relative_to(ROOT)) for p in paths]
        written.extend(rels)
        matrix_rel = f"experiments/oracle_{context_name}.yaml"
        _dump(
            ROOT / matrix_rel,
            {
                "name": f"oracle_{context_name}",
                "description": f"A15 development search 42 cells for {context_name} ER, seed from template.",
                "configs": rels,
            },
        )
        matrices[context_name] = matrix_rel
    for base, algo_path in E04_ALGO_TEMPLATES.items():
        out_dir = ROOT / "configs" / "oracle" / f"core50_nc_{base}_dev"
        paths = emit_configs(ROOT / algo_path, out_dir)
        rels = [str(p.relative_to(ROOT)) for p in paths]
        written.extend(rels)
        matrix_rel = f"experiments/oracle_core50_nc_{base}.yaml"
        _dump(
            ROOT / matrix_rel,
            {
                "name": f"oracle_core50_nc_{base}",
                "description": f"A15 development search 42 cells for CORe50-NC {base}.",
                "configs": rels,
            },
        )
        matrices[f"core50_nc_{base}"] = matrix_rel
    assert len(grid_cells()) == 42
    return {"n_written": len(written), "n_contexts": 8, "n_per_context": 42, "matrices": matrices}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--oracle-search", action="store_true")
    parser.add_argument("--legacy-full-scope", action="store_true", help="Explicitly regenerate superseded full-scope matrices")
    args = parser.parse_args(argv)
    if not args.legacy_full_scope:
        parser.error("Full-scope generator is inactive; use python -m orion_repro.light24 for the current study")
    report: dict[str, Any] = {}
    if args.online:
        report["online"] = emit_online()
    if args.oracle_search:
        report["oracle_search"] = emit_oracle_search()
    if not report:
        parser.error("specify --online and/or --oracle-search")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
