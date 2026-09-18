"""T01 environment checks: orion interpreter, CUDA train step, no mineru."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


FORBIDDEN_PATH_PARTS = ("mineru",)


def _fail(msg: str) -> None:
    print(f"ENVCHECK FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _conda_prefix(exe: Path) -> str | None:
    env = os.environ.get("CONDA_PREFIX")
    if env:
        return env
    # Direct interpreter path: .../envs/orion/bin/python
    if exe.parent.name == "bin" and exe.parent.parent.name == "orion":
        return str(exe.parent.parent)
    return None


def cuda_sm_tag(capability: tuple[int, int]) -> str:
    """PyTorch arch_list tag for a CUDA compute capability, e.g. (11, 0) -> sm_110."""
    major, minor = capability
    return f"sm_{major}{minor}"


def wheel_covers_device(arch_list: list[str], capability: tuple[int, int]) -> bool:
    return cuda_sm_tag(capability) in arch_list


def parse_smi_memory_mib(raw: str) -> int | None:
    """nvidia-smi discrete VRAM; Jetson unified memory often reports N/A."""
    text = raw.strip().strip("[]").replace("MiB", "").strip()
    if not text or text.upper() in {"N/A", "NOT SUPPORTED"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _nvidia_smi_gpus() -> list[dict[str, str | int | None]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"ENVCHECK WARN: nvidia-smi unavailable ({exc})", file=sys.stderr)
        return []
    rows: list[dict[str, str | int | None]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_mib_raw": parts[2],
                    "memory_mib": parse_smi_memory_mib(parts[2]),
                }
            )
    return rows


def verify_training_step(model, opt, x, y) -> None:
    """An optimizer call alone is not evidence that parameters actually changed."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    if not torch.isfinite(loss).all():
        _fail("non-finite training loss")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        _fail("parameter grads missing or non-finite after backward")
    opt.step()
    if any(not torch.isfinite(p).all() for p in params):
        _fail("non-finite parameters after optimizer update")
    if not any(not torch.equal(old, p.detach()) for old, p in zip(before, params)):
        _fail("optimizer did not change any parameter")


def required_avalanche_version() -> str:
    try:
        import avalanche
        from avalanche.training import AGEM, GEM, GSS_greedy, Replay
    except Exception as exc:
        _fail(f"required Avalanche import failed: {exc}")
    print(f"avalanche={avalanche.__version__}")
    print(f"strategies={Replay.__name__},{GSS_greedy.__name__},{GEM.__name__},{AGEM.__name__}")
    return avalanche.__version__


def main() -> None:
    exe = Path(sys.executable).resolve()
    print(f"python={exe}")
    print(f"version={sys.version}")
    if any(part == "mineru" or "envs/mineru" in str(exe) for part in exe.parts):
        _fail(f"interpreter is mineru: {exe}")
    if "orion" not in str(exe):
        _fail(f"interpreter is not the orion env: {exe}")
    if "/home/admin/" in str(exe) or exe.as_posix().startswith("/home/admin/"):
        _fail(f"interpreter is the archived WSL path: {exe}")

    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover
        _fail(f"import torch/torchvision failed: {exc}")

    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False")

    n_gpus = int(torch.cuda.device_count())
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    name = gpu_names[0]
    cap = torch.cuda.get_device_capability(0)
    arch = list(torch.cuda.get_arch_list())
    needed = cuda_sm_tag(cap)
    print(f"n_gpus={n_gpus}")
    print(f"gpus={gpu_names}")
    print(f"capability={cap}")
    print(f"arch_list={arch}")
    print(f"machine={platform.machine()}")
    if not wheel_covers_device(arch, cap):
        _fail(
            f"wheel missing {needed} (arch_list={arch}); "
            f"device {name} capability {cap[0]}.{cap[1]} needs a matching CUDA wheel"
        )
    props = torch.cuda.get_device_properties(0)
    torch_total_mib = int(props.total_memory / (1024 * 1024))
    print(f"torch_total_memory_mib={torch_total_mib}")

    smi = _nvidia_smi_gpus()
    driver = smi[0]["driver_version"] if smi else None
    smi_vram_mib = smi[0].get("memory_mib") if smi else None
    if driver:
        print(f"driver={driver}")
    if smi_vram_mib is None:
        print(
            "ENVCHECK WARN: nvidia-smi discrete VRAM unavailable; "
            "treat torch total_memory as unified-memory accounting, not a discrete VRAM pool",
            file=sys.stderr,
        )

    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    verify_training_step(model, opt, x, y)
    torch.cuda.synchronize(device)
    print("forward_backward_update=ok")

    avalanche_version = required_avalanche_version()

    payload = {
        "executable": str(exe),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "n_gpus": n_gpus,
        "gpu": name,
        "gpus": gpu_names,
        "driver_version": driver,
        "nvidia_smi": smi,
        "nvidia_smi_vram_mib": smi_vram_mib,
        "torch_total_memory_mib": torch_total_mib,
        "memory_model": "unified" if smi_vram_mib is None else "discrete_vram",
        "capability": list(cap),
        "required_sm": needed,
        "arch_list": list(arch),
        "avalanche": avalanche_version,
        "conda_prefix": _conda_prefix(exe),
    }
    out = Path("reports") / "envcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("ENVCHECK PASS")


if __name__ == "__main__":
    main()
