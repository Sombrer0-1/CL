"""Profile intercept/slope memory costs on this GPU (A13)."""

from __future__ import annotations

import json
from pathlib import Path

import psutil
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from orion_repro.memory.cost_model import MemoryCostModel, fit_batch_slope
from orion_repro.models.resnet20 import resnet20

ROOT = Path(__file__).resolve().parents[2]


def _peak_after_step(model, opt, batch: int, device: torch.device) -> int:
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch,), device=device)
    opt.zero_grad(set_to_none=True)
    loss = CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated(device))


def _peak_merged_replay(
    model, opt, new_batch: int, replay_batch: int, device: torch.device
) -> int:
    """Same step, but activations come from cat(new, replay) rather than one tensor."""
    torch.cuda.reset_peak_memory_stats(device)
    x_new = torch.randn(new_batch, 3, 32, 32, device=device)
    x_mem = torch.randn(replay_batch, 3, 32, 32, device=device)
    x = torch.cat([x_new, x_mem], dim=0)
    y = torch.randint(0, 10, (x.size(0),), device=device)
    opt.zero_grad(set_to_none=True)
    loss = CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated(device))


def _plugin_delta_bytes(model, device: torch.device) -> int:
    """Paired default vs GEM patterns + EWC copies, not a full plugin training loop."""
    torch.cuda.synchronize()
    before = int(torch.cuda.memory_allocated(device))
    n_params = sum(p.numel() for p in model.parameters())
    ewc_fisher = torch.zeros(n_params, device=device)
    ewc_saved = torch.zeros(n_params, device=device)
    gem_patterns = torch.zeros(50, 3, 32, 32, device=device)
    torch.cuda.synchronize()
    after = int(torch.cuda.memory_allocated(device))
    delta = max(0, after - before)
    del ewc_fisher, ewc_saved, gem_patterns
    torch.cuda.empty_cache()
    return delta


def _rss() -> int:
    return int(psutil.Process().memory_info().rss)


def _loop_device_peaks(device: torch.device) -> list[tuple[int, int, int]]:
    """Actual train-step peaks for (new, replay) pairs, not the admission helper."""
    model = resnet20(num_classes=10).to(device)
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    pairs = [(8, 8), (16, 16), (32, 32), (16, 0), (64, 16)]
    out = []
    for new_b, replay_b in pairs:
        effective = new_b + replay_b
        if replay_b > 0:
            peak = _peak_merged_replay(model, opt, new_b, replay_b, device)
        else:
            peak = _peak_after_step(model, opt, new_b, device)
        out.append((new_b, replay_b, peak))
        del effective
    return out


def _host_frame_pairs() -> list[tuple[int, int]]:
    """RSS delta vs allocated uint8 RGB frames. Host RSS pairing is noisy; recorded as error evidence."""
    base = _rss()
    held: list[tuple[int, int]] = []
    blobs = []
    for n in (100, 500, 2000):
        blobs.append(torch.zeros(n, 3, 32, 32, dtype=torch.uint8))
        held.append((n, max(0, _rss() - base)))
    del blobs
    return held


def profile(num_classes: int = 10) -> MemoryCostModel:
    if not torch.cuda.is_available():
        raise RuntimeError("cost profiling requires CUDA")
    device = torch.device("cuda")
    model = resnet20(num_classes=num_classes).to(device)
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    _peak_after_step(model, opt, 8, device)
    sizes = [8, 16, 32, 64]
    peaks = [_peak_after_step(model, opt, b, device) for b in sizes]
    intercept, slope, r2 = fit_batch_slope(sizes, peaks)
    merged = _peak_merged_replay(model, opt, 16, 16, device)
    single = _peak_after_step(model, opt, 32, device)
    plugin_delta = _plugin_delta_bytes(model, device)
    loop_pairs = _loop_device_peaks(device)
    fit_x = [n + r for n, r, _ in loop_pairs[:3]]
    fit_y = [p for _, _, p in loop_pairs[:3]]
    loop_intercept, loop_slope, loop_r2 = fit_batch_slope(fit_x, fit_y)
    held = loop_pairs[3:]
    abs_pct = []
    for new_b, replay_b, measured in held:
        pred = loop_intercept + loop_slope * float(new_b + replay_b)
        if measured > 0:
            abs_pct.append(abs(pred - measured) / measured)
    device_mape = float(sum(abs_pct) / len(abs_pct)) if abs_pct else None
    host_pairs = _host_frame_pairs()
    frame = 3 * 32 * 32
    host_err = [abs(n * frame - rss) for n, rss in host_pairs]
    host_mae = float(sum(host_err) / len(host_err)) if host_err else None
    x = torch.zeros(1, 32, 16, 16)
    m_frame = float(x.numel() * 4 + 8)
    uint8_raw = float(frame)
    complete = device_mape is not None and host_mae is not None
    notes = (
        "Device slope from real train-step peaks on cat(new,replay) pairs "
        f"{loop_pairs[:3]}; held-out pairs {held} MAPE={device_mape}. "
        f"Host uint8 frame RSS pairing {host_pairs} MAE={host_mae}B vs {frame}B/frame "
        "(host RSS pairing is noisy; this is not a cgroup cap). "
        f"Synthetic single-tensor slope r2={r2:.4f} vs loop r2={loop_r2:.4f}. "
        f"cat(16,16)={merged} vs single32={single}. "
        f"plugin_gem_ewc_bytes={plugin_delta} still allocation copies, not a full "
        "GEM/EWC training loop. Device slope uses the PyTorch CUDA allocator; on "
        "Jetson unified memory that is not a discrete VRAM pool or cgroup cap."
    )
    return MemoryCostModel(
        intercept_bytes=loop_intercept,
        m_batch_bytes=max(loop_slope, 0.0),
        m_frame_bytes=m_frame,
        plugin_gem_ewc_bytes=float(plugin_delta),
        r2=loop_r2,
        m_frame_raw_uint8_bytes=uint8_raw,
        m_frame_latent_f32_bytes=m_frame,
        resource_for_slope="device",
        complete=complete,
        error_device_mape=device_mape,
        error_host_frame_mae_bytes=host_mae,
        n_device_pairs=len(loop_pairs),
        n_host_pairs=len(host_pairs),
        notes=notes,
    )


def main() -> None:
    model = profile()
    out = ROOT / "data" / "processed" / "memory_cost_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = model.as_dict()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
