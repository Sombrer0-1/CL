"""Standalone Thor cgroup feasibility probe; does not import frozen project code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhuzetong/miniconda3/envs/orion/bin/python'


def snapshot(path):
    out = {'time': time.time()}
    for name in ['memory.current', 'memory.peak', 'memory.max', 'memory.swap.max',
                 'memory.events', 'memory.stat', 'cgroup.procs']:
        try:
            out[name] = (path / name).read_text().strip()
        except OSError:
            pass
    return out


def worker(mode):
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    def record(label, **extra):
        print(json.dumps({'label': label, 'pid': os.getpid(), **snapshot(cg), **extra}), flush=True)
        time.sleep(0.15)
    record('start')
    if mode == 'cache':
        with tempfile.TemporaryFile(dir=ROOT) as f:
            block = bytes(4 * 1024**2)
            for i in range(96):
                f.write(block)
                if i % 16 == 15:
                    f.flush()
                    record('file_write', file_bytes=(i + 1) * len(block))
            os.fsync(f.fileno())
            record('cache_done')
        return
    if mode == 'child':
        result = subprocess.run([PYTHON, str(Path(__file__).resolve()), 'worker', 'cpu'])
        record('child_exit', returncode=result.returncode)
        return
    if mode in ('cuda', 'pinned', 'train'):
        import torch
        torch.cuda.init()
        record('cuda_initialized', torch=torch.__version__)
        if mode == 'train':
            model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10)).cuda()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(64, 512, device='cuda')
            y = torch.randint(10, (64,), device='cuda')
            before = next(model.parameters()).detach().clone()
            for _ in range(5):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            record('training_done', changed=bool((before != next(model.parameters())).any().item()),
                   allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            return
    items = []
    count, chunk = (12, 256 * 1024**2) if mode in ('cuda', 'pinned') else (24, 16 * 1024**2)
    for i in range(count):
        if mode == 'cuda':
            item = torch.empty(chunk, dtype=torch.uint8, device='cuda')
            item.fill_(17)
            torch.cuda.synchronize()
        elif mode == 'pinned':
            item = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
            item.fill_(17)
        else:
            item = bytearray(chunk)
        items.append(item)
        extra = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()} if mode == 'cuda' else {}
        record('allocation', payload_bytes=(i + 1) * chunk, **extra)
    record('completed')


def main():
    results = []
    cases = [('cuda', '512M'), ('cache', '128M')] if '--extra' in sys.argv else [('cpu', '128M'), ('child', '128M'), ('cuda', '2G'), ('pinned', '2G'), ('train', '2G')]
    for mode, quota in cases:
        label = f'{mode}_{quota}' if '--extra' in sys.argv else mode
        unit = f'orion-host-probe-{mode}-{os.getpid()}'
        log = ROOT / f'{label}.log'
        command = ['systemd-run', '--user', '--wait', '--pipe', f'--unit={unit}',
                   '-p', f'MemoryMax={quota}', '-p', 'MemorySwapMax=0',
                   '-p', 'MemoryAccounting=yes', '-p', 'OOMPolicy=stop',
                   PYTHON, str(Path(__file__).resolve()), 'worker', mode]
        samples = []
        cg = Path('/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice') / (unit + '.service')
        with log.open('w') as output:
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                samples.append(snapshot(cg))
                time.sleep(0.02)
            samples.append(snapshot(cg))
        (ROOT / f'{label}_samples.json').write_text(json.dumps(samples, indent=2))
        result = {'mode': mode, 'quota': quota, 'command': command, 'returncode': proc.returncode,
                  'peak_sampled_bytes': max((int(s.get('memory.current', 0)) for s in samples), default=0),
                  'last_events': next((s['memory.events'] for s in reversed(samples) if 'memory.events' in s), None)}
        results.append(result)
        print(json.dumps(result), flush=True)
        subprocess.run(['systemctl', '--user', 'reset-failed', unit + '.service'], capture_output=True)
    (ROOT / ('summary_extra.json' if '--extra' in sys.argv else 'summary.json')).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker(sys.argv[2])
    else:
        main()
