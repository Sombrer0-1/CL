"""effectiveness_v3 constants. Independent of light24 and pressure_v2 defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path

STUDY_ID = "effectiveness_v3"
DESIGN_VERSION = "2026-09-16-v1"
FORMAL_PROTOCOL_ID = "effectiveness_v3_paper_feedback_v1"
DEVELOPMENT_PROTOCOL_ID = "effectiveness_v3_development_v1"
DEFAULT_REVISION = "r1"
SEEDS = (0, 1, 2)
PRIMARY_DATASETS = ("core50_nc", "splitcifar100")
DEVICE = "cuda:0"


def orion_python() -> Path:
    """Resolve the orion interpreter without baking in a host-specific path.

    Order: ORION_PY / ORION_PYTHON, then ~/.conda/envs/orion, then sys.executable.
    execute() still rejects mineru and the archived WSL interpreter.
    """
    explicit = os.environ.get("ORION_PY") or os.environ.get("ORION_PYTHON")
    if explicit:
        return Path(explicit)
    home_conda = Path.home() / ".conda" / "envs" / "orion" / "bin" / "python"
    if home_conda.exists():
        return home_conda
    return Path(sys.executable)
SLOT_COUNTS = {"A": 48, "B": 30, "C": 15, "D": 9, "E": 12, "F": 6, "G": 24, "H": 9}
TOTAL_SLOTS = 153
CORE_SLOTS = 120
SENSITIVITY_SLOTS = 24
HOST_SLOTS = 9
STATIC_GRID = ((16, 200), (16, 2000), (64, 200), (64, 2000), (256, 200), (256, 2000))
EVAL_BATCH_CANDIDATES = (128, 32, 8)
QUOTA_GRID_MIB = 16
RESERVATION_GRID_MIB = 4
PLUGIN_DEFAULTS = {
    "patterns_per_exp": 50,
    "memory_strength": 0.5,
    "ewc_lambda": 100.0,
}
PREFERENCE_ORDERS = {
    "balanced": None,
    "latency": ["latency", "memory", "plasticity", "stability"],
    "plasticity_stability": ["plasticity", "stability", "memory", "latency"],
    "memory": ["memory", "latency", "stability", "plasticity"],
}
GROUP_METHODS = {
    "A": ("S0", "S_star", "O00", "O10", "O01", "O11", "R11", "F11"),
    "B": ("S0", "S_star", "O11", "R11", "F11"),
    "C": ("S0", "S_star_dynamic", "O11", "R11", "O11_half_life"),
    "D": ("O11",),
    "E": ("static_off", "static_on", "orion_off", "orion_on"),
    "F": ("agem_static", "agem_adaptive_ewc"),
    "G": ("O11",),
    "H": ("S0", "S_star_host", "orion_host"),
}
GROUP_VARIANTS = {
    "D": ("latency", "plasticity_stability", "memory"),
    "G": (
        "thr_half",
        "thr_double",
        "alpha_double",
        "beta_double",
        "lr_half",
        "initial_batch32",
        "initial_replay1000",
        "quota_mid",
    ),
}
PROTECTED_PREFIXES = (
    "reports/light24",
    "reports/pressure_v2",
    "configs/light24",
    "configs/pressure_v2",
    "configs/formal",
    "configs/oracle",
    "configs/development",
    "experiments/light24",
    "experiments/pressure_v2",
    "data/manifests",
    "docs/plans/light24_v1.md",
    "docs/protocols",
)
ALLOWED_WRITE_PREFIXES = (
    "configs/effectiveness_v3",
    "experiments/effectiveness_v3/revisions",
    "experiments/registry.jsonl",
    "runs/effectiveness_v3",
    "reports/effectiveness_v3",
    "runs/.orion_project.lock",
    "docs/effectiveness_v3_acceptance.csv",
    "STATUS.md",
)
HISTORICAL_GIT_PATHS = (
    "reports/light24",
    "reports/pressure_v2",
    "configs/light24",
    "configs/pressure_v2",
    "experiments/light24",
    "experiments/pressure_v2",
    "data/manifests",
)
