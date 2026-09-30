"""Generate an additional extlinux LABEL. Does not write /boot unless explicitly applied."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import EXTLINUX_PATH, MEM64_LABEL, MEM64_MENU
from orion_repro.stages.fullmem_v4.util import StageError

PRIMARY_LABEL = "primary"


def parse_extlinux(text: str) -> dict[str, Any]:
    default = None
    timeout = None
    labels: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.upper().startswith("DEFAULT "):
            default = stripped.split(None, 1)[1].strip()
            continue
        if stripped.upper().startswith("TIMEOUT "):
            timeout = stripped.split(None, 1)[1].strip()
            continue
        if stripped.upper().startswith("LABEL "):
            current = stripped.split(None, 1)[1].strip()
            labels[current] = {"raw_name": current}
            continue
        if current is None:
            continue
        key, _, value = stripped.partition(" ")
        labels[current][key.upper()] = value.strip()
    return {"default": default, "timeout": timeout, "labels": labels, "text": text}


def _primary_append(parsed: dict[str, Any]) -> str:
    primary = parsed["labels"].get(PRIMARY_LABEL)
    if not primary or "APPEND" not in primary:
        raise StageError("extlinux.conf has no LABEL primary APPEND; refuse to invent boot args")
    return primary["APPEND"]


def proposed_mem_label(
    original_text: str,
    *,
    mem: str,
    label: str,
    menu: str,
) -> str:
    parsed = parse_extlinux(original_text)
    if PRIMARY_LABEL not in parsed["labels"]:
        raise StageError("missing LABEL primary; refuse to modify boot config")
    append = _primary_append(parsed)
    if re.search(r"(?:^|\s)mem=", append):
        raise StageError("primary APPEND already contains mem=; inspect manually")
    extra_append = f"{append} mem={mem}"
    if label in parsed["labels"]:
        existing = parsed["labels"][label].get("APPEND", "")
        if existing != extra_append:
            raise StageError(f"LABEL {label} exists with different APPEND; refuse to clobber")
    linux = parsed["labels"][PRIMARY_LABEL].get("LINUX") or "/boot/Image"
    initrd = parsed["labels"][PRIMARY_LABEL].get("INITRD") or "/boot/initrd"
    lines = original_text.splitlines()
    out: list[str] = []
    replaced_default = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("DEFAULT "):
            out.append(f"DEFAULT {label}")
            replaced_default = True
            continue
        out.append(line)
    if not replaced_default:
        raise StageError("no DEFAULT line in extlinux.conf")
    if label not in parsed["labels"]:
        if out and out[-1].strip():
            out.append("")
        out.extend(
            [
                f"LABEL {label}",
                f"      MENU LABEL {menu}",
                f"      LINUX {linux}",
                f"      INITRD {initrd}",
                f"      APPEND {extra_append}",
                "",
            ]
        )
    return "\n".join(out).rstrip() + "\n"


def proposed_mem64(original_text: str, *, mem: str = "64G", label: str = MEM64_LABEL) -> str:
    menu = MEM64_MENU if mem == "64G" and label == MEM64_LABEL else f"Orion fullmem_v4 mem={mem}"
    return proposed_mem_label(original_text, mem=mem, label=label, menu=menu)


def rollback_to_primary(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().upper().startswith("DEFAULT "):
            lines.append(f"DEFAULT {PRIMARY_LABEL}")
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def read_live_extlinux(path: Path = EXTLINUX_PATH) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StageError(f"cannot read {path}: {exc}") from exc
