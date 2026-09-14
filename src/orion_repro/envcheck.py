"""T01 environment checks: orion interpreter, CUDA train step, no mineru."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


FORBIDDEN_PATH_PARTS = ("mineru",)


def _fail(msg: str) -> None:
    print(f"ENVCHECK FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    exe = Path(sys.executable).resolve()
    print(f"python={exe}")
    print(f"version={sys.version}")
    if any(part == "mineru" or "envs/mineru" in str(exe) for part in exe.parts):
        _fail(f"interpreter is mineru: {exe}")
    if "orion" not in str(exe):
        _fail(f"interpreter is not the orion env: {exe}")

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

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    print(f"gpu={name}")
    print(f"capability={cap}")
    print(f"arch_list={arch}")
    if cap[0] < 12:
        print("ENVCHECK WARN: capability < 12.0; sm_120 kernels may be missing", file=sys.stderr)

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
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    if any(p.grad is None for p in model.parameters() if p.requires_grad):
        _fail("parameter grads missing after backward")
    print("forward_backward_update=ok")

    try:
        import avalanche
        from avalanche.training import AGEM, GEM, GSS_greedy, Replay

        print(f"avalanche={avalanche.__version__}")
        print(f"strategies={Replay.__name__},{GSS_greedy.__name__},{GEM.__name__},{AGEM.__name__}")
    except Exception as exc:
        print(f"avalanche=unavailable ({exc})")

    payload = {
        "executable": str(exe),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": name,
        "capability": list(cap),
        "arch_list": list(arch),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    out = Path("reports") / "envcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("ENVCHECK PASS")


if __name__ == "__main__":
    main()
