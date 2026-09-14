"""Admission / stop budget guard. Not a substitute for URGE (PLAN §8.3)."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetAdmissionError(RuntimeError):
    """Enforcement requested but the cost model/limit contract is not satisfied."""


@dataclass
class BudgetDecision:
    accepted: bool
    reason: str
    suggested_new_batch: int
    suggested_replay: int
    applied_new_batch: int
    applied_replay: int


def formula_stop_if_infeasible(
    *,
    suggested_new_batch: int,
    suggested_replay: int,
    predicted_bytes: int | None,
    limit_bytes: int | None,
    min_batch: int,
    max_batch: int | None = None,
) -> BudgetDecision:
    """orion_formula: no silent clipping; reject if the raw proposal is illegal or over limit."""
    if suggested_new_batch < min_batch:
        return BudgetDecision(
            False,
            "batch_below_min",
            suggested_new_batch,
            suggested_replay,
            suggested_new_batch,
            suggested_replay,
        )
    if max_batch is not None and suggested_new_batch > int(max_batch):
        return BudgetDecision(
            False,
            "batch_above_max",
            suggested_new_batch,
            suggested_replay,
            suggested_new_batch,
            suggested_replay,
        )
    if suggested_replay < 0:
        return BudgetDecision(
            False,
            "replay_negative",
            suggested_new_batch,
            suggested_replay,
            suggested_new_batch,
            suggested_replay,
        )
    if limit_bytes is not None and predicted_bytes is None:
        return BudgetDecision(
            False,
            "cost_model_required",
            suggested_new_batch,
            suggested_replay,
            suggested_new_batch,
            suggested_replay,
        )
    if limit_bytes is not None and predicted_bytes is not None and predicted_bytes > limit_bytes:
        return BudgetDecision(
            False,
            "preflight_infeasible",
            suggested_new_batch,
            suggested_replay,
            suggested_new_batch,
            suggested_replay,
        )
    return BudgetDecision(
        True,
        "ok",
        suggested_new_batch,
        suggested_replay,
        suggested_new_batch,
        suggested_replay,
    )


def guarded_clip(
    *,
    suggested_new_batch: int,
    suggested_replay: int,
    predicted_bytes: int,
    limit_bytes: int,
    min_batch: int,
    bytes_per_batch_step: int,
    bytes_per_replay_item: int,
    max_batch: int | None = None,
) -> BudgetDecision:
    """orion_guarded engineering extension: clip down until predicted_bytes <= limit."""
    batch = suggested_new_batch
    replay = suggested_replay
    if max_batch is not None:
        batch = min(batch, int(max_batch))
    reason = "ok"
    while batch >= min_batch:
        pred = predicted_bytes
        pred = max(0, predicted_bytes - (suggested_new_batch - batch) * bytes_per_batch_step)
        pred = max(0, pred - (suggested_replay - replay) * bytes_per_replay_item)
        if pred <= limit_bytes:
            if batch != suggested_new_batch or replay != suggested_replay:
                reason = "clipped"
            return BudgetDecision(True, reason, suggested_new_batch, suggested_replay, batch, replay)
        if replay > 0:
            replay -= max(1, replay // 10)
        elif batch > min_batch:
            batch -= 1
        else:
            break
    return BudgetDecision(
        False,
        "cannot_clip_to_limit",
        suggested_new_batch,
        suggested_replay,
        batch,
        replay,
    )


def require_cost_model_for_enforcement(enforcement: str, cost_model) -> None:
    if enforcement in {"formula_stop", "guarded_clip", "host_enforced", "device_allocator_enforced"}:
        if cost_model is None:
            raise BudgetAdmissionError(
                f"budget.enforcement={enforcement} requires a cost model; "
                "cannot claim a memory budget is in effect"
            )
