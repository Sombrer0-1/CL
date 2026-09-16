"""Host memory enforcement probe and optional delegated-cgroup executor."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any


class HostCapabilityError(RuntimeError):
    pass


def cgroup_memory_max() -> dict[str, Any]:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return {
                "available": False,
                "path": str(path),
                "reason": f"unreadable: {exc}",
                "mechanism": "cgroup_memory_max",
            }
        writable = os.access(path, os.W_OK)
        return {
            "available": raw not in {"max", ""} and raw.isdigit(),
            "writable": writable,
            "path": str(path),
            "value": raw,
            "mechanism": "cgroup_memory_max",
            "reason": None
            if writable
            else "process cannot write cgroup memory.max; host_enforced blocked",
        }
    return {
        "available": False,
        "writable": False,
        "path": None,
        "mechanism": "cgroup_memory_max",
        "reason": "no cgroup memory.max in this namespace",
    }


def rlimit_as_supported() -> dict[str, Any]:
    try:
        import resource
    except ImportError:
        return {
            "available": False,
            "mechanism": "rlimit_as",
            "reason": "resource module unavailable",
        }
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "available": True,
        "mechanism": "rlimit_as",
        "soft": None if soft == resource.RLIM_INFINITY else int(soft),
        "hard": None if hard == resource.RLIM_INFINITY else int(hard),
        "scope": "process virtual address space, not RSS; CUDA mappings inflate VA",
        "reason": "usable as a named host_rlimit_as variant, not a cgroup RSS cap",
    }


def probe_host_enforcement() -> dict[str, Any]:
    cgroup = cgroup_memory_max()
    rlimit = rlimit_as_supported()
    host_enforced = bool(cgroup.get("writable"))
    return {
        "host_enforced": "available" if host_enforced else "blocked",
        "cgroup": cgroup,
        "rlimit_as": rlimit,
        "notes": (
            "PLAN host_enforced requires a verified process-group/cgroup limit. "
            "This probe never writes host/global memory settings. "
            "Writable parent memory.max is not an execution proof; HostEnvelope "
            "must create a child cgroup without modifying the parent limit."
        ),
    }


def probe_delegated_cgroup(delegated_root: Path | None = None) -> dict[str, Any]:
    """Check whether a child cgroup can be created. Never writes parent memory.max."""
    root = Path(delegated_root) if delegated_root is not None else None
    if root is None:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "no delegated cgroup root configured",
            "mechanism": "cgroup_v2_child",
        }
    if not root.exists() or not root.is_dir():
        return {
            "available": False,
            "status": "unavailable",
            "reason": f"delegated root missing: {root}",
            "mechanism": "cgroup_v2_child",
        }
    controllers = root / "cgroup.controllers"
    subtree = root / "cgroup.subtree_control"
    memory_max = root / "memory.max"
    can_mkdir = os.access(root, os.W_OK | os.X_OK)
    return {
        "available": bool(can_mkdir and (root / "cgroup.procs").exists()),
        "status": "available" if can_mkdir else "unavailable",
        "path": str(root),
        "writable": can_mkdir,
        "has_controllers": controllers.exists(),
        "has_subtree_control": subtree.exists(),
        "parent_memory_max_present": memory_max.exists(),
        "parent_memory_max_writable": os.access(memory_max, os.W_OK) if memory_max.exists() else False,
        "mechanism": "cgroup_v2_child",
        "reason": None if can_mkdir else "cannot create child directory under delegated root",
        "notes": "parent memory.max must not be modified even if writable",
    }


class HostEnvelope:
    """Experiment-owned child cgroup. Close only removes an empty group we created."""

    def __init__(
        self,
        path: Path,
        quota_bytes: int,
        swap_policy: str,
        *,
        created: bool,
    ) -> None:
        self.path = path
        self.quota_bytes = int(quota_bytes)
        self.swap_policy = str(swap_policy)
        self.created = bool(created)
        self.attached_pids: list[int] = []

    @classmethod
    def create(
        cls,
        delegated_root: Path | None,
        quota: int,
        swap_policy: str = "max",
    ) -> "HostEnvelope":
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
            raise HostCapabilityError("quota must be a positive integer byte count")
        probe = probe_delegated_cgroup(delegated_root)
        if not probe.get("available"):
            raise HostCapabilityError(probe.get("reason") or "delegated cgroup unavailable")
        root = Path(delegated_root)
        name = f"orion_h_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        path = root / name
        path.mkdir(exist_ok=False)
        try:
            memory_max = path / "memory.max"
            if memory_max.exists() or True:
                memory_max.write_text(str(int(quota)), encoding="utf-8")
            swap = path / "memory.swap.max"
            if swap_policy in {"disable", "0"}:
                try:
                    swap.write_text("0", encoding="utf-8")
                except OSError as exc:
                    raise HostCapabilityError(f"unable to set memory.swap.max: {exc}") from exc
        except OSError as exc:
            try:
                path.rmdir()
            except OSError:
                pass
            raise HostCapabilityError(f"unable to configure child cgroup: {exc}") from exc
        return cls(path, quota, swap_policy, created=True)

    def attach_before_exec(self, pid: int) -> None:
        procs = self.path / "cgroup.procs"
        procs.write_text(str(int(pid)), encoding="utf-8")
        self.attached_pids.append(int(pid))

    def read_events(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": str(self.path), "quota_bytes": self.quota_bytes}
        for name in ("memory.current", "memory.peak", "memory.events", "memory.swap.max", "memory.max"):
            path = self.path / name
            if not path.exists():
                payload[name] = None
                continue
            try:
                payload[name] = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                payload[name] = f"unreadable:{exc}"
        return payload

    def close(self) -> None:
        if not self.created:
            return
        try:
            self.path.rmdir()
        except OSError:
            # Non-empty or still occupied; leave the group rather than touching parents.
            return

