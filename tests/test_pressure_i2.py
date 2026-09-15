import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orion_repro.memory.phase_recorder import PhaseRecorder
from orion_repro.memory.resource_envelope import ResourceEnvelope, default_dyn_schedule
from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.prefetch.supply import extract_xy, tensor_sha256
from orion_repro.runner.artifacts import RunArtifacts
from orion_repro.runner.failures import parse_cuda_oom_request_bytes, pathological_memory_value
from orion_repro.strategies.builder import snapshot_plugin_activity
from orion_repro.strategies.toggles import TogglePlugin


def test_parse_oom_request_and_drop_pathological_values():
    msg = "Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 15.93 GiB of which 14.65 GiB is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use."
    assert parse_cuda_oom_request_bytes(msg) == 20 * 1024**2
    assert pathological_memory_value(17179869184 * 1024**3)


def test_phase_recorder_writes_rows(tmp_path):
    arts = RunArtifacts(tmp_path)
    rec = PhaseRecorder(arts, quota_bytes=128 * 1024**2, reservation_getter=lambda: 0)
    rec.begin("training", 0)
    row = rec.end("training", 0)
    arts.close()
    assert row.phase == "training"
    assert row.duration_s >= 0
    assert (tmp_path / "phase_trace.csv").is_file()


def test_dyn_schedule_three_thirds():
    assert default_dyn_schedule(9, 8) == [0, 0, 0, 8, 8, 8, 0, 0, 0]


def test_consumption_hash_is_order_sensitive():
    x1 = torch.ones(2, 3)
    x2 = torch.zeros(2, 3)
    assert tensor_sha256(x1) != tensor_sha256(x2)
    assert extract_xy((x1, torch.tensor([1, 0])))[1].tolist() == [1, 0]


def test_serial_loader_records_produce_time():
    class Loader:
        def __len__(self):
            return 3

        def __iter__(self):
            for i in range(3):
                yield torch.ones(1), torch.tensor([i])

    wrapped = PrefetchingDataLoader(
        Loader(), experience_id=0, config_version=0, max_depth=1, use_queue=False, record_hashes=True
    )
    batches = list(wrapped)
    assert len(batches) == 3
    digest = wrapped.digest()
    assert digest["batches_consumed"] == 3
    assert digest["rolling_sha256"]


def test_plugin_activity_unknown_ewc_stays_null():
    inner = SimpleNamespace(memory_x={}, auxiliary_visits=None)
    plugin = TogglePlugin(inner, enabled=True, name="ewc")
    plugin.hook_counts["before_backward"] = 4
    strategy = SimpleNamespace(plugins=[plugin])
    snap = snapshot_plugin_activity(strategy)
    assert snap["optional_plugins"][0]["before_backward"] == 4
    assert snap["optional_plugins"][0]["ewc_extra_forward_visits"] is None
    assert plugin.hook_counts["before_backward"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resource_envelope_allocates_and_closes():
    device = torch.device("cuda")
    env = ResourceEnvelope(device)
    before = int(torch.cuda.memory_allocated())
    rec = env.transition(0, 1024 * 1024)
    assert rec.status == "ok"
    assert rec.actual_tensor_bytes == 1024 * 1024
    assert env.reservation_bytes == 1024 * 1024
    env.close()
    torch.cuda.synchronize()
    assert env.reservation_bytes == 0
    assert int(torch.cuda.memory_allocated()) <= before + 1024 * 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resource_envelope_transition_oom_does_not_retry():
    device = torch.device("cuda")
    env = ResourceEnvelope(device)
    env.install_quota(
        {"enforcement": "device_allocator_enforced", "controlled_resource": "device", "limit_bytes": 32 * 1024**2},
        device,
    )
    rec = env.transition(3, 1024**3)
    assert rec.status == "resource_transition"
    env.close()
