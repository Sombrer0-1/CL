from __future__ import annotations

import math
from dataclasses import dataclass

from orion_repro.control.urge import (
    UrgeConfig,
    scale_budget,
    select_optimizer_mode,
    threshold,
    urge_factors,
)


@dataclass
class ControlState:
    mb: float
    mr: float
    new_batch: int
    replay_capacity: int
    optimizer_mode: str


@dataclass
class ControlDecision:
    t: int
    factors: dict[str, float]
    urge: float
    thr: float
    mb_next: float
    mr_next: float
    suggested_new_batch: int
    suggested_replay: int
    suggested_optimizer_mode: str
    applied: ControlState | None = None
    clip_reason: str | None = None


def map_budget_to_counts(
    mb: float,
    mr: float,
    *,
    m_batch: float,
    m_frame: float,
) -> tuple[int, int]:
    """Raw floor mapping of Eq. (3)–(4). Does not apply min/max engineering clips.

    PLAN §8.3: orion_formula keeps the unclipped integer proposal so the guard
    can reject batch_below_min / batch_above_max explicitly.
    """
    if m_batch <= 0 or m_frame <= 0:
        raise ValueError("m_batch and m_frame must be positive")
    batch = int(math.floor(mb / m_batch))
    replay = int(math.floor(mr / m_frame))
    return batch, replay


def step_controller(
    cfg: UrgeConfig,
    state: ControlState,
    *,
    t: int,
    plasticity: float,
    stability: float,
    latency_s: float,
    memory_mib: float,
    m_batch: float,
    m_frame: float,
) -> ControlDecision:
    factors = urge_factors(
        plasticity,
        stability,
        latency_s,
        memory_mib,
        kp=cfg.kp,
        ks=cfg.ks,
        kl=cfg.kl,
        km=cfg.km,
        p_th=cfg.p_th,
        s_th=cfg.s_th,
        latency_th_s=cfg.latency_th_s,
        m_max_mib=cfg.m_max_mib,
    )
    thr = threshold(cfg.thr0, cfg.delta, t)
    urge = factors["urge"]
    mb_next = scale_budget(state.mb, cfg.alpha, urge, thr)
    mr_next = scale_budget(state.mr, cfg.beta, urge, thr)
    mode = select_optimizer_mode(urge, thr, equal_uses_gt=cfg.equal_uses_gt)
    batch, replay = map_budget_to_counts(
        mb_next,
        mr_next,
        m_batch=m_batch,
        m_frame=m_frame,
    )
    return ControlDecision(
        t=t,
        factors=factors,
        urge=urge,
        thr=thr,
        mb_next=mb_next,
        mr_next=mr_next,
        suggested_new_batch=batch,
        suggested_replay=replay,
        suggested_optimizer_mode=mode,
    )
