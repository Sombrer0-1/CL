from orion_repro.control.urge import preference_weights, scale_budget, stable_sigmoid, urge_factors


def test_sigmoid_at_zero():
    assert abs(stable_sigmoid(0.0) - 0.5) < 1e-15


def test_urge_at_thresholds_is_one_sixteenth():
    out = urge_factors(
        0.4,
        0.7,
        1.5,
        100.0,
        kp=0.25,
        ks=0.25,
        kl=0.25,
        km=0.25,
        p_th=0.4,
        s_th=0.7,
        latency_th_s=1.5,
        m_max_mib=100.0,
    )
    assert abs(out["factor_p"] - 0.5) < 1e-12
    assert abs(out["factor_s"] - 0.5) < 1e-12
    assert abs(out["factor_l"] - 0.5) < 1e-12
    assert abs(out["factor_m"] - 0.5) < 1e-12
    assert abs(out["urge"] - 0.0625) < 1e-12


def test_increasing_memory_decreases_urge():
    kwargs = dict(
        plasticity=0.4,
        stability=0.7,
        latency_s=1.5,
        kp=0.25,
        ks=0.25,
        kl=0.25,
        km=0.25,
        p_th=0.4,
        s_th=0.7,
        latency_th_s=1.5,
        m_max_mib=100.0,
    )
    low = urge_factors(memory_mib=80.0, **kwargs)["urge"]
    high = urge_factors(memory_mib=120.0, **kwargs)["urge"]
    assert high < low


def test_budget_keeps_fractional_residual():
    nxt = scale_budget(100.0, 0.1, 0.125, 0.1)
    assert abs(nxt - 100.25) < 1e-12


def test_preference_example_from_paper():
    w = preference_weights(["memory", "plasticity", "stability", "latency"])
    assert abs(w["memory"] - 0.4) < 1e-12
    assert abs(w["plasticity"] - 0.3) < 1e-12
    assert abs(w["stability"] - 0.2) < 1e-12
    assert abs(w["latency"] - 0.1) < 1e-12


def test_resolve_coefficients_ranked_overrides_yaml():
    from orion_repro.control.urge import resolve_controller_coefficients

    coef = resolve_controller_coefficients(
        {
            "preference": "ranked",
            "preference_order": ["memory", "latency", "plasticity", "stability"],
            "coefficients": {"kp": 0.25, "ks": 0.25, "kl": 0.25, "km": 0.25},
        }
    )
    assert abs(coef["km"] - 0.4) < 1e-12
    assert abs(coef["kl"] - 0.3) < 1e-12
    assert abs(coef["kp"] - 0.2) < 1e-12
    assert abs(coef["ks"] - 0.1) < 1e-12


def test_resolve_coefficients_balanced():
    from orion_repro.control.urge import resolve_controller_coefficients

    coef = resolve_controller_coefficients({"preference": "balanced"})
    assert coef == {"kp": 0.25, "ks": 0.25, "kl": 0.25, "km": 0.25}
