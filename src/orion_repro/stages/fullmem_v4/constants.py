"""Identity and path constants for fullmem_v4. Do not reuse v3 frozen hashes."""

from __future__ import annotations

from pathlib import Path

STUDY_ID = "fullmem_v4"
DESIGN_VERSION = "design-v1"
ORION_PYTHON = "/home/zhuzetong/miniconda3/envs/orion/bin/python"
PROJECT_LOCK_REL = "runs/.orion_project.lock"
EXECUTOR_UNIT = "orion-fullmem-v4-executor.service"
TASK_UNIT_PREFIX = "orion-v4-task-"
MIN_FREE_BYTES = 10 * 1024 * 1024 * 1024
IDLE_SLEEP_S = 2.0
HEARTBEAT_S = 5.0
DEFAULT_CAPACITY = "unrestricted"
MEM64_LABEL = "orion-mem64g"
MEM64_MENU = "Orion fullmem_v4 mem=64G"
EXTLINUX_PATH = Path("/boot/extlinux/extlinux.conf")

ALLOWED_WRITE_PREFIXES = (
    "configs/fullmem_v4/",
    "experiments/fullmem_v4/",
    "reports/fullmem_v4/",
    "runs/fullmem_v4/",
    "runs/.orion_project.lock",
)

PROTECTED_PREFIXES = (
    "configs/effectiveness_v3/",
    "configs/light24/",
    "configs/pressure_v2/",
    "experiments/effectiveness_v3/",
    "reports/effectiveness_v3/",
    "docs/history/",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
