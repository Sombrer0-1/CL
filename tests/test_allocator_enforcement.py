from types import SimpleNamespace
import pytest
import torch
from orion_repro.memory.enforcement import install_device_quota


def test_cuda_without_index_resolved_and_invalid_budget_rejected(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.cuda,'current_device',lambda: 0)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda index: SimpleNamespace(total_memory=1024))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda fraction,index: calls.append((fraction,index)))
    budget={'enforcement':'device_allocator_enforced','controlled_resource':'device','limit_bytes':256}
    install_device_quota(budget,torch.device('cuda'))
    assert calls==[(0.25,0)]
    with pytest.raises(ValueError):install_device_quota({**budget,'limit_bytes':2048},torch.device('cuda'))
    with pytest.raises(ValueError):install_device_quota(budget,torch.device('cpu'))
    assert len(calls)==1
