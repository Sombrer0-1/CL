"""Resumable representative study with estimates, never a training time cutoff."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

from orion_repro.runner.locks import acquire_locks
from orion_repro.runner.spec import load_yaml, validate_mapping
from orion_repro.runner.reuse import find_completed, reuse_identity

ROOT = Path(__file__).resolve().parents[2]
TARGET_SECONDS = 24 * 3600


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)


def remaining_seconds(state, limit=TARGET_SECONDS):
    return max(0.0, float(limit) - float(state.get('consumed_s', 0)))


def reconcile_active(state):
    """Keep interrupted attempts visible when a parent was lost."""
    active = state.get('active')
    if not active:
        return
    pid = active.get('pid')
    if pid and Path(f'/proc/{pid}').exists():
        raise RuntimeError(f'Unresolved prior child PID {pid}; inspect before resuming')
    # A child may have completed just before the parent lost its accounting.
    final = ROOT/'runs'/active['run_id']/'summary.json'
    outcome = json.loads(final.read_text()).get('status', 'interrupted') if final.exists() else 'interrupted'
    state['results'].append({**active, 'status': outcome, 'elapsed_s': None, 'accounting': 'unknown'})
    if not final.exists():
        append_outcome(active['run_id'], 'interrupted', 'Executor lost; elapsed time unknown')
    state['active'] = None


def stop_group(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def append_outcome(run_id, status, reason):
    with (ROOT/'experiments/registry.jsonl').open('a') as f:
        f.write(json.dumps({'run_id': run_id, 'status': status, 'reason': reason,
                            'reconciled_at': datetime.now(timezone.utc).isoformat()})+'\n')


def execute(path, spec, identity, state, state_path):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_light24_'+uuid.uuid4().hex[:8]
    out = ROOT/'runs'/'_light24_executor'
    out.mkdir(parents=True, exist_ok=True)
    cfg = out/f'{run_id}.json'
    cfg.write_text(json.dumps({**spec, 'run_id': run_id}))
    active = dict(config=str(path.relative_to(ROOT)), identity=identity, run_id=run_id,
                  pid=None)
    state['active'] = active
    save(state_path, state)
    start = time.monotonic()
    status = 'implementation_error'
    proc = None
    interrupted = False
    try:
        with (out/f'{run_id}.log').open('w') as log:
            env = dict(os.environ, WANDB_MODE='disabled')
            if spec.get('training', {}).get('deterministic_algorithms'):
                env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            proc = subprocess.Popen([sys.executable, '-m', 'orion_repro.run', '--config', str(cfg)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            active['pid'] = proc.pid
            save(state_path, state)
            code = proc.wait()
            summary_path = ROOT/'runs'/run_id/'summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            status = summary.get('status', 'implementation_error')
            if code != 0 and status == 'completed':
                status = 'implementation_error' 
    except KeyboardInterrupt:
        interrupted = True
        if proc:
            stop_group(proc)
        status = 'interrupted'
        append_outcome(run_id, status, 'Study interrupted by signal')
    finally:
        if proc:
            stop_group(proc)
        elapsed = time.monotonic()-start
        state['consumed_s'] += elapsed
        row = {**active, 'status': status, 'elapsed_s': elapsed}
        state['results'].append(row)
        state['active'] = None
        save(state_path, state)
    print(json.dumps(row), flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', default='experiments/light24/core.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    matrix = load_yaml(ROOT/args.matrix)
    if matrix.get('study_id') != 'light24_v1':
        parser.error('Only approved light24_v1 matrices are accepted')
    paths = [ROOT/p for p in matrix['configs']]
    specs = [load_yaml(p) for p in paths]
    for spec in specs:
        validate_mapping(spec, require_provenance=False)
        if spec.get('study_id') != 'light24_v1' or spec.get('reuse_version') != 2:
            parser.error('Study configs must opt into light24_v1 and reuse v2')
        if not reuse_identity(spec, ROOT):
            parser.error('Frozen dataset manifest required')
    state_path = ROOT/'runs/light24_progress.json'
    try:
        locks = acquire_locks([ROOT/'runs/.orion_project.lock', ROOT/'runs/.light24.lock'])
    except BlockingIOError:
        parser.error('Another Orion training executor is active')
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'study_id': 'light24_v1', 'consumed_s': 0, 'active': None, 'results': []}
        if args.dry_run:
            print(json.dumps({'matrix': args.matrix, 'n_configs': len(paths),
                              'target_remaining_s': remaining_seconds(state),
                              'estimated_hours': matrix.get('estimated_hours'), 'active_run': state.get('active'),
                              'note': 'Estimates only; no time cutoff. Verify GPU is free before execution'}, indent=2))
            return
        reconcile_active(state)
        save(state_path, state)
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        for path, spec in zip(paths, specs):
            identity = reuse_identity(spec, ROOT)
            previous = find_completed(ROOT, identity)
            if previous:
                print(json.dumps({'config': str(path), 'status': 'reused', 'run_id': previous['run_id']}), flush=True)
                continue
            terminal = next((x for x in reversed(state['results']) if x['identity'] == identity and x['status'] in {
                'cuda_oom', 'host_oom', 'budget_exceeded', 'preflight_infeasible'}), None)
            if terminal:
                print(json.dumps({'config': str(path), 'status': 'previous_failure_preserved', 'outcome': terminal['status']}), flush=True)
                continue
            print(json.dumps({'starting': str(path), 'consumed_s': state['consumed_s']}), flush=True)
            status = execute(path, spec, identity, state, state_path)
            if status == 'implementation_error':
                print(json.dumps({'paused': True, 'reason': status}), flush=True)
                break
        print(json.dumps({'consumed_s': state['consumed_s'], 'target_remaining_s': remaining_seconds(state)}))
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
