"""Boot and platform identity for true shared-memory admission."""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.constants import STUDY_ID
from orion_repro.stages.fullmem_v4.util import utc_now


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    text = _read_text("/proc/meminfo") or ""
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        num = raw.strip().split()[0]
        try:
            out[key] = int(num)
        except ValueError:
            continue
    return out


def _nvpmodel() -> dict[str, str]:
    try:
        proc = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"query": "unavailable"}
    lines = (proc.stdout or "").strip().splitlines()
    name = lines[0] if lines else ""
    mode = lines[1] if len(lines) > 1 else ""
    return {"query": (proc.stdout or "").strip(), "name": name, "mode": mode, "returncode": str(proc.returncode)}


def parse_mem_cmdline(cmdline: str) -> str | None:
    match = re.search(r"(?:^|\s)mem=([0-9]+[KkMmGg]?)\b", cmdline)
    return match.group(1) if match else None


def classify_capacity(cmdline: str, mem_total_kb: int) -> dict[str, Any]:
    requested = parse_mem_cmdline(cmdline)
    if requested is None:
        identity = "unrestricted"
    else:
        identity = f"mem{requested.lower()}"
    return {
        "capacity_id": identity,
        "requested_mem": requested,
        "mem_total_kb": mem_total_kb,
        "mem_total_gib_approx": round(mem_total_kb / (1024 * 1024), 3),
    }


def collect_identity(*, include_cuda: bool = False) -> dict[str, Any]:
    mem = _meminfo()
    cmdline = _read_text("/proc/cmdline") or ""
    mem_total = int(mem.get("MemTotal") or 0)
    capacity = classify_capacity(cmdline, mem_total)
    uname = {}
    if hasattr(os, "uname"):
        raw = os.uname()
        uname = {
            "sysname": raw.sysname,
            "nodename": raw.nodename,
            "release": raw.release,
            "version": raw.version,
            "machine": raw.machine,
        }
    payload: dict[str, Any] = {
        "study_id": STUDY_ID,
        "collected_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "boot_id": _read_text("/proc/sys/kernel/random/boot_id"),
        "cmdline": cmdline,
        "uname": uname,
        "meminfo_kb": {
            "MemTotal": mem.get("MemTotal"),
            "MemFree": mem.get("MemFree"),
            "MemAvailable": mem.get("MemAvailable"),
            "Cached": mem.get("Cached"),
            "Buffers": mem.get("Buffers"),
            "SwapTotal": mem.get("SwapTotal"),
            "SwapFree": mem.get("SwapFree"),
        },
        "swap_is_zero": int(mem.get("SwapTotal") or 0) == 0,
        "capacity": capacity,
        "nvpmodel": _nvpmodel(),
        "pid": os.getpid(),
    }
    if include_cuda:
        payload["cuda"] = _cuda_identity()
    return payload


def _cuda_identity() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment specific
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "cuda_unavailable"}
    props = torch.cuda.get_device_properties(0)
    return {
        "ok": True,
        "device_name": props.name,
        "total_memory": int(props.total_memory),
        "major": props.major,
        "minor": props.minor,
        "torch": torch.__version__,
    }


def capacity_matches(required: str, current: dict[str, Any]) -> bool:
    if required in {"any", "", "unrestricted-or-any"}:
        return True
    current_id = str((current.get("capacity") or {}).get("capacity_id") or "")
    if required == "unrestricted":
        return current_id == "unrestricted"
    return current_id == required
