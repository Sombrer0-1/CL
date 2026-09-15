from types import SimpleNamespace
from orion_repro.memory.probe import _pss_or_none, _resolve_cuda_device


def test_pss_zero_is_measured_not_unavailable():
    assert _pss_or_none(SimpleNamespace(memory_full_info=lambda: SimpleNamespace(pss=0))) == 0
    assert _pss_or_none(SimpleNamespace(memory_full_info=lambda: SimpleNamespace())) is None


def test_resolve_cuda_device_indexes_requested_gpu():
    import torch

    if not torch.cuda.is_available():
        return
    d0 = _resolve_cuda_device("cuda:0")
    assert d0 is not None and d0.index == 0
    if torch.cuda.device_count() > 1:
        d1 = _resolve_cuda_device("cuda:1")
        assert d1 is not None and d1.index == 1

