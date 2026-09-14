from types import SimpleNamespace
from orion_repro.memory.probe import _pss_or_none


def test_pss_zero_is_measured_not_unavailable():
    assert _pss_or_none(SimpleNamespace(memory_full_info=lambda: SimpleNamespace(pss=0))) == 0
    assert _pss_or_none(SimpleNamespace(memory_full_info=lambda: SimpleNamespace())) is None
