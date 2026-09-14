from orion_repro.control.controller import ControlState, map_budget_to_counts, step_controller
from orion_repro.control.urge import UrgeConfig, select_optimizer_mode
from orion_repro.memory.budget import formula_stop_if_infeasible


def test_equal_urge_uses_algorithm1_gt():
    assert select_optimizer_mode(0.5, 0.5, equal_uses_gt=True) == "default"
    assert select_optimizer_mode(0.5, 0.5, equal_uses_gt=False) == "advanced"


def test_next_batch_from_float_budget():
    cfg = UrgeConfig(
        kp=0.25,
        ks=0.25,
        kl=0.25,
        km=0.25,
        p_th=0.5,
        s_th=0.5,
        latency_th_s=1.0,
        m_max_mib=100.0,
        thr0=0.1,
        delta=0.0,
        alpha=0.1,
        beta=0.2,
    )
    state = ControlState(mb=16.0, mr=200.0, new_batch=16, replay_capacity=200, optimizer_mode="default")
    decision = step_controller(
        cfg,
        state,
        t=0,
        plasticity=0.5,
        stability=0.5,
        latency_s=1.0,
        memory_mib=100.0,
        m_batch=1.0,
        m_frame=1.0,
    )
    # U=0.0625, Thr=0.1, MB=16*(1+0.1*(0.0625-0.1))=15.94 -> floor 15
    assert abs(decision.mb_next - 15.94) < 1e-10
    assert decision.suggested_new_batch == 15
    assert decision.suggested_optimizer_mode == "default"


def test_map_budget_keeps_raw_floor_zero():
    """MB=0 must propose batch=0, not min_batch."""
    batch, replay = map_budget_to_counts(0.0, 3.9, m_batch=1.0, m_frame=1.0)
    assert batch == 0
    assert replay == 3


def test_formula_stop_sees_batch_below_min():
    d = formula_stop_if_infeasible(
        suggested_new_batch=0,
        suggested_replay=10,
        predicted_bytes=1,
        limit_bytes=100,
        min_batch=1,
    )
    assert d.accepted is False
    assert d.reason == "batch_below_min"
    assert d.suggested_new_batch == 0


def test_formula_stop_sees_batch_above_max():
    d = formula_stop_if_infeasible(
        suggested_new_batch=2048,
        suggested_replay=10,
        predicted_bytes=1,
        limit_bytes=10**12,
        min_batch=1,
        max_batch=1024,
    )
    assert d.accepted is False
    assert d.reason == "batch_above_max"


def test_formula_stop_requires_cost_model_when_limited():
    d = formula_stop_if_infeasible(
        suggested_new_batch=16,
        suggested_replay=200,
        predicted_bytes=None,
        limit_bytes=1,
        min_batch=1,
    )
    assert d.accepted is False
    assert d.reason == "cost_model_required"
