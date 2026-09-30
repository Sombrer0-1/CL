"""Dedicated H4 background occupancy process.

PLAN §2.3 / design-v1 §4.2: hold, touch, and release real pages in a
separate process. Do not fake pressure by rewriting controller M_max.
Do not enlarge the model. Do not exhaust the board.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


SEQUENCES = ("low_high_low", "high_low_high")
CHUNK = 4096


class H4HolderError(RuntimeError):
    """Holder exited, timed out, or failed to keep the requested occupancy."""


def meminfo_kb() -> dict[str, int]:
    out: dict[str, int] = {}
    text = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        token = rest.strip().split()[0]
        try:
            out[key] = int(token)
        except ValueError:
            continue
    return out


def memtotal_mib() -> float:
    return float(meminfo_kb().get("MemTotal") or 0) / 1024.0


def memavailable_bytes() -> int:
    return int(meminfo_kb().get("MemAvailable") or 0) * 1024


def h4_reservation_schedule(
    n_experiences: int,
    *,
    high_bytes: int,
    sequence: str,
    low_bytes: int = 0,
) -> list[int]:
    """Switch at floor(N/3) and floor(2N/3). Experience-index, not wall-clock."""
    if n_experiences <= 0:
        raise ValueError("n_experiences must be positive")
    if sequence not in SEQUENCES:
        raise ValueError(f"unknown H4 sequence {sequence!r}")
    high_bytes = int(high_bytes)
    low_bytes = int(low_bytes)
    if high_bytes < 0 or low_bytes < 0:
        raise ValueError("hold bytes must be >= 0")
    first = n_experiences // 3
    second = (2 * n_experiences) // 3
    out: list[int] = []
    for i in range(n_experiences):
        if i < first:
            high_phase = sequence == "high_low_high"
        elif i < second:
            high_phase = sequence == "low_high_low"
        else:
            high_phase = sequence == "high_low_high"
        out.append(high_bytes if high_phase else low_bytes)
    return out


def _touch(buf: bytearray) -> None:
    view = memoryview(buf)
    n = len(buf)
    for offset in range(0, n, CHUNK):
        view[offset] = 1
    if n:
        view[-1] = 1


def _rss_kb() -> int:
    try:
        return int(Path("/proc/self/status").read_text(encoding="utf-8").split("VmRSS:")[1].split()[0])
    except (OSError, IndexError):
        return -1


def serve() -> int:
    """JSON-line worker. CPU pages only; no CUDA, no fork-from-trainer."""
    buf = bytearray()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        op = str(msg.get("op") or "")
        if op == "stop":
            buf = bytearray()
            print(json.dumps({"ok": True, "op": "stop", "rss_kb": _rss_kb()}), flush=True)
            return 0
        if op == "ping":
            print(
                json.dumps({"ok": True, "op": "ping", "held_bytes": len(buf), "rss_kb": _rss_kb()}),
                flush=True,
            )
            continue
        if op != "set":
            print(json.dumps({"ok": False, "error": f"unknown_op:{op}"}), flush=True)
            continue
        requested = int(msg.get("bytes") or 0)
        if requested < 0:
            print(json.dumps({"ok": False, "error": "negative_bytes"}), flush=True)
            continue
        buf = bytearray()
        if requested:
            buf = bytearray(requested)
            _touch(buf)
        print(
            json.dumps(
                {
                    "ok": True,
                    "op": "set",
                    "requested": requested,
                    "actual": len(buf),
                    "rss_kb": _rss_kb(),
                }
            ),
            flush=True,
        )
    return 0


@dataclass
class HolderRecord:
    experience_index: int
    requested_bytes: int
    actual_tensor_bytes: int
    allocated_before: int | None
    allocated_after: int | None
    reserved_before: int | None
    reserved_after: int | None
    status: str
    notes: str = ""
    mem_available_before_bytes: int | None = None
    mem_available_after_bytes: int | None = None
    holder_rss_kb: int | None = None
    holder_pid: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackgroundHolder:
    def __init__(self, proc: subprocess.Popen[str], log: TextIO | None = None) -> None:
        self.proc = proc
        self.log = log
        self.reservation_bytes = 0

    @classmethod
    def start(cls, *, log_path: Path | None = None, executable: str | None = None) -> "BackgroundHolder":
        exe = executable or sys.executable
        log = None
        stderr: Any = subprocess.DEVNULL
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            stderr = log
        proc = subprocess.Popen(
            [exe, "-m", "orion_repro.memory.h4_holder", "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        holder = cls(proc, log=log)
        ping = holder._rpc({"op": "ping"}, timeout_s=15)
        if not ping.get("ok"):
            holder.close()
            raise H4HolderError(f"holder_ping_failed:{ping}")
        return holder

    @property
    def pid(self) -> int | None:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _rpc(self, msg: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        if not self.alive():
            raise H4HolderError(f"holder_exited:{self.proc.returncode}")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout_s
        buf = ""
        while time.monotonic() < deadline:
            if not self.alive():
                raise H4HolderError(f"holder_exited:{self.proc.returncode}")
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([self.proc.stdout], [], [], min(1.0, max(0.05, remaining)))
            if not ready:
                continue
            chunk = self.proc.stdout.readline()
            if chunk == "":
                raise H4HolderError("holder_stdout_closed")
            buf += chunk
            if "\n" in buf:
                return json.loads(buf)
        raise H4HolderError(f"holder_rpc_timeout:{msg.get('op')}")

    def ping(self) -> dict[str, Any]:
        return self._rpc({"op": "ping"}, timeout_s=10)

    def transition(self, k: int, reservation_bytes: int) -> HolderRecord:
        requested = int(reservation_bytes)
        before = memavailable_bytes() if Path("/proc/meminfo").is_file() else None
        ping = self.ping()
        timeout_s = 30.0 + max(0, requested) / (64 * 1024 * 1024)
        reply = self._rpc({"op": "set", "bytes": requested}, timeout_s=timeout_s)
        after = memavailable_bytes() if Path("/proc/meminfo").is_file() else None
        if not reply.get("ok"):
            self.reservation_bytes = 0
            return HolderRecord(
                experience_index=int(k),
                requested_bytes=requested,
                actual_tensor_bytes=0,
                allocated_before=None,
                allocated_after=None,
                reserved_before=None,
                reserved_after=None,
                status="resource_transition",
                notes=str(reply.get("error") or "set_failed"),
                mem_available_before_bytes=before,
                mem_available_after_bytes=after,
                holder_rss_kb=reply.get("rss_kb") or ping.get("rss_kb"),
                holder_pid=self.pid,
            )
        actual = int(reply.get("actual") or 0)
        self.reservation_bytes = actual
        notes = "unchanged" if requested == actual and requested == int(ping.get("held_bytes") or -1) else ""
        if actual != requested:
            notes = f"actual_mismatch:{actual}!={requested}"
        return HolderRecord(
            experience_index=int(k),
            requested_bytes=requested,
            actual_tensor_bytes=actual,
            allocated_before=None,
            allocated_after=None,
            reserved_before=None,
            reserved_after=None,
            status="ok",
            notes=notes,
            mem_available_before_bytes=before,
            mem_available_after_bytes=after,
            holder_rss_kb=reply.get("rss_kb"),
            holder_pid=self.pid,
        )

    def close(self) -> None:
        try:
            if self.alive():
                try:
                    self._rpc({"op": "stop"}, timeout_s=10)
                except H4HolderError:
                    pass
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            self.reservation_bytes = 0
            if self.log is not None:
                try:
                    self.log.close()
                except OSError:
                    pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--serve"] or (len(args) == 1 and args[0] == "--serve"):
        return serve()
    print("usage: python -m orion_repro.memory.h4_holder --serve", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
