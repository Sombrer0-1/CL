"""H4 sequence indices. Does not exhaust a board."""

from __future__ import annotations

import pytest

from orion_repro.memory.h4_holder import SEQUENCES, h4_reservation_schedule


def test_nc9_low_high_low_switches_at_3_and_6():
    sched = h4_reservation_schedule(9, high_bytes=100, sequence="low_high_low")
    assert sched == [0, 0, 0, 100, 100, 100, 0, 0, 0]


def test_nc9_high_low_high_switches_at_3_and_6():
    sched = h4_reservation_schedule(9, high_bytes=100, sequence="high_low_high")
    assert sched == [100, 100, 100, 0, 0, 0, 100, 100, 100]


def test_nic79_floor_splits():
    n = 79
    first = n // 3
    second = (2 * n) // 3
    assert first == 26
    assert second == 52
    low = h4_reservation_schedule(n, high_bytes=7, sequence="low_high_low")
    high = h4_reservation_schedule(n, high_bytes=7, sequence="high_low_high")
    assert low[:first] == [0] * first
    assert low[first:second] == [7] * (second - first)
    assert low[second:] == [0] * (n - second)
    assert high[:first] == [7] * first
    assert high[first:second] == [0] * (second - first)
    assert high[second:] == [7] * (n - second)


def test_cifar10_remainder_last_third_keeps_tail():
    # N=10: floor(10/3)=3, floor(20/3)=6, last third has 4 experiences.
    sched = h4_reservation_schedule(10, high_bytes=1, sequence="low_high_low")
    assert sched == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_unknown_sequence_refused():
    with pytest.raises(ValueError):
        h4_reservation_schedule(9, high_bytes=1, sequence="mid_only")
    assert "low_high_low" in SEQUENCES
    assert "high_low_high" in SEQUENCES


def test_systemd_oom_kill_is_resource_failure():
    from orion_repro.stages.fullmem_v4.executor import _classify_exit

    log = (
        "Finished with result: oom-kill\n"
        "Main processes terminated with: code=killed/status=TERM\n"
    )
    assert _classify_exit(1, log) == "resource_failure"
    assert _classify_exit(0, "status: completed") == "completed"
