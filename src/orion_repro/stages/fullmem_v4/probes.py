"""G1 shared-pool probes. Graded growth; never try to kill the whole board."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

from orion_repro.stages.fullmem_v4.identity import collect_identity
from orion_repro.stages.fullmem_v4.queue import ensure_dirs
from orion_repro.stages.fullmem_v4.util import assert_write_path, atomic_write_json, utc_now


def _mem_available_kb() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable missing")


def _touch_bytes(n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    for offset in range(0, n, 4096):
        view[offset] = 1
    if n:
        view[-1] = 1
    return buf


def _snapshot(tag: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"tag": tag, "ts": utc_now(), "mem_available_kb": _mem_available_kb(), "rss_kb": _rss_kb()}
    if extra:
        row.update(extra)
    return row


def _rss_kb() -> int:
    try:
        return int(Path("/proc/self/status").read_text(encoding="utf-8").split("VmRSS:")[1].split()[0])
    except (OSError, IndexError):
        return -1


def run_shared_pool_probe(*, max_gib: float = 8.0, chunk_mib: int = 256, out_dir: Path | None = None) -> dict[str, Any]:
    if max_gib > 16:
        raise ValueError("G1 probes refuse max_gib>16; do not exhaust the board")
    identity = collect_identity(include_cuda=True)
    max_bytes = int(max_gib * (1024**3))
    chunk = int(chunk_mib * (1024**2))
    samples: list[dict[str, Any]] = []
    samples.append(_snapshot("start"))
    result: dict[str, Any] = {
        "probe": "shared_pool_g1",
        "started_at_utc": utc_now(),
        "identity": identity,
        "max_bytes": max_bytes,
        "chunk_bytes": chunk,
        "samples": samples,
        "steps": {},
    }

    cpu_hold = _touch_bytes(min(chunk * 4, max_bytes // 4 or chunk))
    samples.append(_snapshot("cpu_hold", {"held_bytes": len(cpu_hold)}))
    del cpu_hold
    gc.collect()
    time.sleep(0.5)
    samples.append(_snapshot("cpu_released"))
    result["steps"]["cpu_alloc_free"] = "ok"

    cuda_info: dict[str, Any] = {"ok": False}
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        n = min(chunk * 4, max_bytes // 4)
        tensor = torch.empty((n,), device="cuda", dtype=torch.uint8)
        tensor.fill_(1)
        torch.cuda.synchronize()
        samples.append(_snapshot("cuda_hold", {"held_bytes": n, "allocated": int(torch.cuda.memory_allocated())}))
        del tensor
        torch.cuda.synchronize()
        gc.collect()
        time.sleep(0.5)
        samples.append(_snapshot("cuda_released", {"allocated": int(torch.cuda.memory_allocated())}))
        cuda_info = {"ok": True, "bytes": n}
        result["steps"]["cuda_alloc_free"] = "ok"
    except Exception as exc:
        cuda_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["steps"]["cuda_alloc_free"] = "failed"
        samples.append(_snapshot("cuda_failed", {"error": cuda_info["error"]}))
    result["cuda"] = cuda_info

    try:
        import torch

        n = min(chunk * 2, max_bytes // 8)
        pinned = torch.empty((n,), dtype=torch.uint8, pin_memory=True)
        pinned.fill_(1)
        samples.append(_snapshot("pinned_hold", {"held_bytes": n}))
        del pinned
        gc.collect()
        result["steps"]["pinned_alloc_free"] = "ok"
    except Exception as exc:
        result["steps"]["pinned_alloc_free"] = f"{type(exc).__name__}: {exc}"
        samples.append(_snapshot("pinned_failed", {"error": result["steps"]["pinned_alloc_free"]}))

    cache_path = Path("/tmp/orion_fullmem_v4_pagecache.bin")
    try:
        size = 64 * 1024 * 1024
        with cache_path.open("wb") as handle:
            block = bytes(1024 * 1024)
            for _ in range(64):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        samples.append(_snapshot("pagecache_written", {"bytes": size}))
        cache_path.unlink(missing_ok=True)
        result["steps"]["pagecache"] = "ok"
    except Exception as exc:
        result["steps"]["pagecache"] = f"{type(exc).__name__}: {exc}"
        samples.append(_snapshot("pagecache_failed", {"error": result["steps"]["pagecache"]}))

    child_bytes = min(chunk * 2, max_bytes // 8)
    pid = os.fork() if hasattr(os, "fork") else -1
    if pid == 0:
        buf = _touch_bytes(child_bytes)
        time.sleep(1.0)
        os._exit(0 if buf[0] == 1 else 1)
    if pid > 0:
        samples.append(_snapshot("child_alive", {"child_pid": pid, "held_bytes": child_bytes}))
        _, status = os.waitpid(pid, 0)
        samples.append(_snapshot("child_exited", {"wait_status": int(status)}))
        result["steps"]["child"] = "ok" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f"status={status}"
    else:
        result["steps"]["child"] = "fork_unavailable"

    mixed: dict[str, Any] = {}
    try:
        import torch

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.2)
        result["mixed_notes"] = (
            "empty_cache used only to isolate mixed-step measurement from the prior CUDA cache; "
            "not a training-time URGE policy"
        )
        cpu_n = min(chunk * 4, max_bytes // 4)
        gpu_n = min(chunk * 4, max_bytes // 4)
        before = _mem_available_kb()
        gpu = torch.empty((gpu_n,), device="cuda", dtype=torch.uint8)
        gpu.fill_(1)
        torch.cuda.synchronize()
        mid = _mem_available_kb()
        cpu = _touch_bytes(cpu_n)
        both = _mem_available_kb()
        del cpu
        gc.collect()
        after_cpu = _mem_available_kb()
        gpu2_n = min(chunk * 2, max_bytes // 8)
        gpu2 = torch.empty((gpu2_n,), device="cuda", dtype=torch.uint8)
        gpu2.fill_(1)
        torch.cuda.synchronize()
        after_gpu2 = _mem_available_kb()
        mixed = {
            "ok": True,
            "before_avail_kb": before,
            "after_gpu_avail_kb": mid,
            "after_cpu_and_gpu_avail_kb": both,
            "after_cpu_free_avail_kb": after_cpu,
            "after_more_gpu_avail_kb": after_gpu2,
            "gpu_drop_kb": before - mid,
            "cpu_extra_drop_kb": mid - both,
            "cpu_free_recovered_kb": after_cpu - both,
            "more_gpu_drop_kb": after_cpu - after_gpu2,
        }
        del gpu, gpu2
        torch.cuda.synchronize()
        result["steps"]["mixed"] = "ok"
        samples.append(_snapshot("mixed_done", mixed))
    except Exception as exc:
        mixed = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["steps"]["mixed"] = mixed["error"]
        samples.append(_snapshot("mixed_failed", mixed))
    result["mixed"] = mixed
    result["finished_at_utc"] = utc_now()
    samples.append(_snapshot("end"))

    paths = ensure_dirs()
    dest_dir = out_dir or (paths["reports"] / "admission" / time.strftime("%Y%m%dT%H%M%SZ"))
    dest_dir = assert_write_path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "shared_pool.json"
    atomic_write_json(out_path, result)
    result["output"] = str(out_path)
    return result
