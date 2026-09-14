"""Linear host/device cost model from on-device profiling (A13).

Device activation slope and host storage frames are separate. Mixing them
into one prediction is invalid; callers must name the resource.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MemoryCostModel:
    intercept_bytes: float
    m_batch_bytes: float
    m_frame_bytes: float
    plugin_gem_ewc_bytes: float
    r2: float
    notes: str = ""
    m_frame_raw_uint8_bytes: float = 3 * 32 * 32
    m_frame_latent_f32_bytes: float = 32 * 16 * 16 * 4 + 8
    resource_for_slope: str = "device"
    complete: bool = False
    error_device_mape: float | None = None
    error_host_frame_mae_bytes: float | None = None
    n_device_pairs: int = 0
    n_host_pairs: int = 0

    def predict_device_bytes(
        self,
        *,
        new_batch: int,
        replay_batch: int,
        advanced: bool = False,
    ) -> int:
        """GPU allocator model vs effective training batch = new + replay."""
        extra = self.plugin_gem_ewc_bytes if advanced else 0.0
        effective = float(new_batch) + float(max(0, replay_batch))
        return int(round(self.intercept_bytes + self.m_batch_bytes * effective + extra))

    def predict_host_replay_bytes(self, replay_capacity: int, *, representation: str) -> int:
        if representation in {"latent", "latent_f32"}:
            frame = self.m_frame_latent_f32_bytes
        else:
            frame = self.m_frame_raw_uint8_bytes
        return int(round(frame * float(max(0, replay_capacity))))

    def predict_bytes(
        self,
        new_batch: int,
        replay_capacity: int,
        *,
        replay_batch: int | None = None,
        advanced: bool = False,
        resource: str = "device",
        representation: str = "avalanche_buffer",
    ) -> int:
        """Admission helper. Device slope does not add latent host frames."""
        rb = int(new_batch if replay_batch is None else replay_batch)
        if resource == "device":
            return self.predict_device_bytes(
                new_batch=new_batch, replay_batch=rb, advanced=advanced
            )
        if resource == "host":
            return self.predict_host_replay_bytes(replay_capacity, representation=representation)
        raise ValueError(f"unknown resource {resource}")

    def as_dict(self) -> dict:
        return asdict(self)


def fit_batch_slope(batch_sizes: list[int], peak_bytes: list[int]) -> tuple[float, float, float]:
    import numpy as np

    x = np.asarray(batch_sizes, dtype=np.float64)
    y = np.asarray(peak_bytes, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two batch sizes")
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(intercept), float(slope), float(r2)
