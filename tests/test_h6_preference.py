"""H6 preference / threshold-decay unit tests. Not a freeze."""

from __future__ import annotations

import math

import pytest

from orion_repro.control.urge import (
    H6A_PREFERENCE_ORDERS,
    h6_delta,
    preference_weights,
    resolve_controller_coefficients,
    threshold,
)


def test_h6a_orders_are_4_3_2_1():
    lat = preference_weights(H6A_PREFERENCE_ORDERS["latency"])
    assert lat["latency"] == pytest.approx(0.4)
    assert lat["memory"] == pytest.approx(0.3)
    assert lat["plasticity"] == pytest.approx(0.2)
    assert lat["stability"] == pytest.approx(0.1)
    ps = preference_weights(H6A_PREFERENCE_ORDERS["p_s"])
    assert ps["plasticity"] == pytest.approx(0.4)
    assert ps["stability"] == pytest.approx(0.3)
    assert ps["memory"] == pytest.approx(0.2)
    assert ps["latency"] == pytest.approx(0.1)
    mem = preference_weights(H6A_PREFERENCE_ORDERS["memory"])
    assert mem["memory"] == pytest.approx(0.4)
    assert mem["latency"] == pytest.approx(0.3)
    assert mem["stability"] == pytest.approx(0.2)
    assert mem["plasticity"] == pytest.approx(0.1)


def test_named_preference_overrides_coefficients():
    coef = resolve_controller_coefficients(
        {
            "preference": "latency",
            "coefficients": {"kp": 0.25, "ks": 0.25, "kl": 0.25, "km": 0.25},
        }
    )
    assert coef["kl"] == pytest.approx(0.4)
    assert coef["km"] == pytest.approx(0.3)
    bal = resolve_controller_coefficients({"preference": "balanced"})
    assert bal == {"kp": 0.25, "ks": 0.25, "kl": 0.25, "km": 0.25}


def test_h6b_delta_half_life_reconstruction():
    assert h6_delta("delta0", 10) == 0.0
    assert h6_delta("ln2", 10) == pytest.approx(math.log(2.0))
    assert threshold(0.05, h6_delta("ln2", 10), 1) == pytest.approx(0.025)
    n = 10
    d = h6_delta("n-1", n)
    assert d == pytest.approx(math.log(2.0) / 9.0)
    assert threshold(0.05, d, n - 1) == pytest.approx(0.025)
    nic = h6_delta("n-1", 79)
    assert nic == pytest.approx(math.log(2.0) / 78.0)
