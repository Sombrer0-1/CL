"""URGE (Eq. 1) and time-dependent threshold / memory updates (Eq. 2–5).

Main protocol uses the paper product of four sigmoids with raw units.
Weights k_* are the only paper-normalized quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

FLOAT = np.float64


def _require_finite(name: str, value: float | np.floating | int) -> float:
    try:
        parsed = FLOAT(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(parsed)


def stable_sigmoid(x: float | np.floating) -> float:
    """Numerically stable sigmoid in float64; math-equivalent to 1/(1+exp(-x))."""
    z = FLOAT(x)
    if z >= 0.0:
        ez = np.exp(-z)
        return FLOAT(1.0 / (1.0 + ez)).item()
    ez = np.exp(z)
    return FLOAT(ez / (1.0 + ez)).item()


def urge_factors(
    plasticity: float,
    stability: float,
    latency_s: float,
    memory_mib: float,
    *,
    kp: float,
    ks: float,
    kl: float,
    km: float,
    p_th: float,
    s_th: float,
    latency_th_s: float,
    m_max_mib: float,
) -> dict[str, float]:
    """Eq. (1): four factors and their product.

    factor_p = σ(-kp (P - P_th))
    factor_s = σ(-ks (S - S_th))
    factor_l = σ( kl (L - L_th))
    factor_m = σ(-km (M - M_max))
    """
    plasticity = _require_finite("plasticity", plasticity)
    stability = _require_finite("stability", stability)
    latency_s = _require_finite("latency_s", latency_s)
    memory_mib = _require_finite("memory_mib", memory_mib)
    kp = _require_finite("kp", kp)
    ks = _require_finite("ks", ks)
    kl = _require_finite("kl", kl)
    km = _require_finite("km", km)
    p_th = _require_finite("p_th", p_th)
    s_th = _require_finite("s_th", s_th)
    latency_th_s = _require_finite("latency_th_s", latency_th_s)
    m_max_mib = _require_finite("m_max_mib", m_max_mib)
    fp = stable_sigmoid(-FLOAT(kp) * (FLOAT(plasticity) - FLOAT(p_th)))
    fs = stable_sigmoid(-FLOAT(ks) * (FLOAT(stability) - FLOAT(s_th)))
    fl = stable_sigmoid(FLOAT(kl) * (FLOAT(latency_s) - FLOAT(latency_th_s)))
    fm = stable_sigmoid(-FLOAT(km) * (FLOAT(memory_mib) - FLOAT(m_max_mib)))
    product = FLOAT(fp) * FLOAT(fs) * FLOAT(fl) * FLOAT(fm)
    return {
        "factor_p": float(fp),
        "factor_s": float(fs),
        "factor_l": float(fl),
        "factor_m": float(fm),
        "urge": float(product),
    }


def threshold(thr0: float, delta: float, t: int) -> float:
    """Eq. (2): Thr_t = Thr0 * exp(-δ t). t is 0-based completed-experience index."""
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    thr0 = _require_finite("thr0", thr0)
    delta = _require_finite("delta", delta)
    t = int(t)
    return float(FLOAT(thr0) * np.exp(-FLOAT(delta) * FLOAT(t)))


def scale_budget(current: float, coeff: float, urge: float, thr: float) -> float:
    """Eq. (3)/(4) multiplicative update; algebraically same as Alg.1 if/else on MB/MR.

    MB_next = MB * (1 + α (U - Thr)). Does not integer-truncate residual.
    """
    current = _require_finite("current", current)
    coeff = _require_finite("coeff", coeff)
    urge = _require_finite("urge", urge)
    thr = _require_finite("thr", thr)
    nxt = FLOAT(current) * (FLOAT(1.0) + FLOAT(coeff) * (FLOAT(urge) - FLOAT(thr)))
    if not np.isfinite(nxt) or nxt < 0.0:
        raise ValueError(f"illegal next budget {nxt} from current={current} coeff={coeff}")
    return float(nxt)


def select_optimizer_mode(urge: float, thr: float, *, equal_uses_gt: bool = True) -> str:
    """Eq. (5) uses >= ; Algorithm 1 line 7 uses >.

    Default A03: follow Algorithm 1, so equality keeps default plugin.
    """
    urge = _require_finite("urge", urge)
    thr = _require_finite("thr", thr)
    if equal_uses_gt:
        return "advanced" if FLOAT(urge) > FLOAT(thr) else "default"
    return "advanced" if FLOAT(urge) >= FLOAT(thr) else "default"


def preference_weights(order: list[str]) -> dict[str, float]:
    """§4.4: last item weight 1, second-to-last 2, ... then divide by sum.

    `order` is most-important first, matching the paper example
    [memory, plasticity, stability, training latency] -> (4,3,2,1).
    """
    keys = ["memory", "plasticity", "stability", "latency"]
    aliases = {
        "training latency": "latency",
        "p": "plasticity",
        "s": "stability",
        "m": "memory",
        "l": "latency",
    }
    mapped = [aliases.get(item.lower(), item.lower()) for item in order]
    if sorted(mapped) != sorted(keys):
        raise ValueError(f"preference order must be a permutation of {keys}, got {order}")
    n = len(mapped)
    raw = {name: float(n - i) for i, name in enumerate(mapped)}
    total = sum(raw.values())
    return {k: raw[k] / total for k in keys}


def balanced_weights() -> dict[str, float]:
    """§5.2.3 'equal importance' operationalization (not written as 0.25 in the PDF)."""
    return {k: 0.25 for k in ("memory", "plasticity", "stability", "latency")}


def coefficients_from_weights(weights: dict[str, float]) -> dict[str, float]:
    """Map §4.4 preference weights onto Eq. (1) k_p, k_s, k_l, k_m."""
    return {
        "kp": float(weights["plasticity"]),
        "ks": float(weights["stability"]),
        "kl": float(weights["latency"]),
        "km": float(weights["memory"]),
    }


def resolve_controller_coefficients(controller: dict) -> dict[str, float]:
    """Return kp/ks/kl/km. Ranked preference_order overrides explicit coefficients."""
    pref = controller.get("preference")
    order = controller.get("preference_order")
    if pref == "balanced":
        return coefficients_from_weights(balanced_weights())
    if order:
        return coefficients_from_weights(preference_weights(list(order)))
    coef = controller.get("coefficients") or {}
    missing = [k for k in ("kp", "ks", "kl", "km") if k not in coef]
    if missing:
        raise ValueError(f"controller coefficients missing {missing}")
    return {k: float(coef[k]) for k in ("kp", "ks", "kl", "km")}


@dataclass(frozen=True)
class UrgeConfig:
    kp: float
    ks: float
    kl: float
    km: float
    p_th: float
    s_th: float
    latency_th_s: float
    m_max_mib: float
    thr0: float
    delta: float
    alpha: float
    beta: float
    equal_uses_gt: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping) -> "UrgeConfig":
        return cls(
            kp=float(data["kp"]),
            ks=float(data["ks"]),
            kl=float(data["kl"]),
            km=float(data["km"]),
            p_th=float(data["p_th"]),
            s_th=float(data["s_th"]),
            latency_th_s=float(data["latency_th_s"]),
            m_max_mib=float(data["m_max_mib"]),
            thr0=float(data["thr0"]),
            delta=float(data["delta"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            equal_uses_gt=bool(data.get("equal_uses_gt", True)),
        )


def bytes_to_mib(n_bytes: int | float) -> float:
    return float(FLOAT(n_bytes) / FLOAT(1024.0 * 1024.0))
