"""AGEM group-balanced loader must not emit batch_size=0 on long streams."""

from orion_repro.strategies.agem_compat import agem_group_batch_size


def test_agem_group_batch_size_matches_short_stream():
    assert agem_group_batch_size(64, 10) == 6
    assert agem_group_batch_size(64, 9) == 7
    assert agem_group_batch_size(64, 64) == 1


def test_agem_group_batch_size_nic_does_not_collapse():
    # Avalanche 0.6.0 does 64 // 65 == 0 and crashes at NIC experience 65.
    assert agem_group_batch_size(64, 65) == 1
    assert agem_group_batch_size(64, 79) == 1
    assert agem_group_batch_size(64, 0) == 0
