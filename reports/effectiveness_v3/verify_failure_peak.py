"""Small real CUDA OOM check; allocator-only diagnostic, not a learning experiment.

Run from the repository root with the orion interpreter. Changes only this
process's CUDA allocator quota; writes validation_gpu.json beside this script.
"""
from pathlib import Path
import json
import tempfile

import torch

from orion_repro.memory.enforcement import install_device_quota
from orion_repro.memory.phase_recorder import PhaseRecorder
from orion_repro.runner.artifacts import RunArtifacts


def main():
    device = torch.device('cuda:0')
    mib = 1024 ** 2
    quota = install_device_quota({
        'enforcement': 'device_allocator_enforced',
        'controlled_resource': 'device', 'limit_bytes': 64 * mib,
    }, device)
    with tempfile.TemporaryDirectory(prefix='orion_phase_audit_') as tmp:
        arts = RunArtifacts(Path(tmp))
        recorder = PhaseRecorder(arts, quota_bytes=64 * mib,
                                 reservation_getter=lambda: 0, device=device)
        recorder.begin('training', 0)
        small = torch.empty(2 * mib, device=device, dtype=torch.uint8)
        small.fill_(1)
        trained = recorder.end('training', 0)
        del small
        recorder.begin('evaluation', 0)
        larger = torch.empty(6 * mib, device=device, dtype=torch.uint8)
        larger.fill_(1)
        try:
            torch.empty(128 * mib, device=device, dtype=torch.uint8)
        except torch.cuda.OutOfMemoryError:
            failed = recorder.end_failed('evaluation', 0)
        else:
            raise AssertionError('expected allocator quota OOM')
        assert failed is not None and failed.notes == 'failed'
        assert failed.allocated_peak_bytes > trained.allocated_peak_bytes
        assert failed.allocated_peak_bytes >= 6 * mib
        payload = {'kind': 'allocator_phase_diagnostic', 'passed': True,
                   'device': str(device), 'gpu': torch.cuda.get_device_name(device),
                   'torch': torch.__version__, 'quota': quota,
                   'previous_phase': trained.as_row(), 'failed_phase': failed.as_row(),
                   'limitations': 'Real allocations and OOM; no benchmark training or effectiveness evidence.'}
        arts.close()
    out = Path(__file__).with_name('validation_gpu.json')
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'PASS: failed evaluation peak {failed.allocated_peak_bytes} > previous phase {trained.allocated_peak_bytes}; {out}')


if __name__ == '__main__':
    main()
