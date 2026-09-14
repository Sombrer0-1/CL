"""Emit the approved representative study and select static baselines on development data."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import yaml
from orion_repro.runner.spec import load_yaml, validate_mapping
from orion_repro.runner.reuse import find_completed, reuse_identity

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = ('splitcifar100', 'core50_nc')
SEEDS = (0, 1, 2)
GRID = [(b, r) for b in (16, 64, 256) for r in (200, 2000)]
# Planning allowances, not deadlines. Includes startup/evaluation and uncertainty.
ESTIMATES = {'core': [0.5, 2], 'search': [0.5, 2], 'selected': [0.25, 1],
             'budget': [0.5, 2], 'preference': [0.25, 1], 'prefetch': [0.25, 1],
             'sensitivity': [0.5, 2], 'endless': [1, 4]}


def base(context, seed, method='orion_formula'):
    spec = load_yaml(ROOT/f'configs/formal/e03/{context}/orion_formula/seed{seed}.yaml')
    spec.update(study_id='light24_v1', reuse_version=2, protocol_id='light24_paper_feedback_v1',
                alignment_version='light24_v1', method_id=method)
    spec['controller']['plugin_policy'] = 'adaptive'
    if method == 'er_static':
        spec['controller']['enabled'] = False
    elif method == 'orion_batch_replay':
        spec['controller']['plugin_policy'] = 'fixed_default'
    return spec


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def emit(root=ROOT):
    groups = {k: [] for k in ESTIMATES if k != 'selected'}
    def add(group, name, spec):
        spec = copy.deepcopy(spec)
        spec.update(experiment_id=f'L_{group}', claim_ids=[f'L_{group}'])
        validate_mapping(spec, require_provenance=False)
        rel = f'configs/light24/{group}/{name}.yaml'
        write(root/rel, spec)
        groups[group].append(rel)
    for ds in CONTEXTS:
        for seed in SEEDS:
            for method in ('er_static', 'orion_batch_replay', 'orion_formula'):
                add('core', f'{ds}_{method}_s{seed}', base(ds, seed, method))
            for thr in (.025, .1):
                spec = base(ds, seed)
                spec['controller']['thr0'] = thr
                add('sensitivity', f'{ds}_thr{thr}_s{seed}', spec)
        template = 'cifar100' if ds == 'splitcifar100' else ds
        dev = load_yaml(ROOT/f'configs/development/{template}_er_static.yaml')
        dev.update(study_id='light24_v1', reuse_version=2, protocol_id='light24_development_v1',
                   alignment_version='light24_v1')
        for batch, replay in GRID:
            spec = copy.deepcopy(dev)
            spec['training'].update(new_batch=batch, replay_batch=batch)
            spec['replay']['capacity'] = replay
            add('search', f'{ds}_b{batch}_r{replay}', spec)
    for seed in SEEDS:
        for limit in (128, 256, 512):
            for method in ('er_static', 'orion_formula'):
                spec = base('splitcifar100', seed, method)
                spec['budget'].update(enforcement='device_allocator_enforced', limit_bytes=limit*1024**2)
                spec['controller']['thresholds']['m_max_mib'] = float(limit)
                add('budget', f'cifar100_{method}_{limit}mib_s{seed}', spec)
        for pref, order in {'latency': ['latency','memory','plasticity','stability'],
                            'ps': ['plasticity','stability','memory','latency']}.items():
            spec = base('core50_nc', seed)
            spec['controller'].update(preference='ranked', preference_order=order)
            add('preference', f'core50_nc_{pref}_s{seed}', spec)
        for enabled in (False, True):
            spec = base('core50_nc', seed, 'er_static')
            spec['training']['deterministic_algorithms'] = True
            spec['prefetch'].update(enabled=enabled, queue_depth=2)
            add('prefetch', f'core50_nc_{"on" if enabled else "off"}_s{seed}', spec)
        for scenario in ('ic', 'il', 'wc'):
            for method in ('er_static', 'orion_formula'):
                spec = load_yaml(ROOT/f'configs/development/endless_{scenario}_{method}_grouped_v2.yaml')
                spec.update(study_id='light24_v1', reuse_version=2,
                            protocol_id='light24_endless_grouped_v1', alignment_version='light24_v1')
                spec['seeds'] = dict.fromkeys(('model','stream','replay','augmentation'), seed)
                spec['dataset']['split_manifest'] = 'data/manifests/endless_cl_sim.json'
                add('endless', f'endless_{scenario}_{method}_s{seed}', spec)
    for name, paths in groups.items():
        write(root/f'experiments/light24/{name}.yaml', dict(study_id='light24_v1', name=name,
              estimated_hours=ESTIMATES[name], configs=paths))
    order = ['core', 'search', 'budget', 'preference', 'prefetch', 'sensitivity', 'endless']
    write(root/'experiments/light24/all.yaml', dict(study_id='light24_v1', name='all_static_stages',
          estimated_hours=[4,14], configs=[p for name in order for p in groups[name]]))
    write(root/'experiments/light24/extensions.yaml', dict(study_id='light24_v1', name='extensions',
          estimated_hours=[2.5,10], configs=[p for name in order[2:] for p in groups[name]]))
    print(json.dumps({k:len(v) for k,v in groups.items()}))


def select_static(root=ROOT):
    matrix = load_yaml(root/'experiments/light24/search.yaml')
    grouped = {}
    for rel in matrix['configs']:
        spec = load_yaml(root/rel)
        if spec['phase'] != 'development' or spec['controller']['feedback_source'] != 'development_val_seen':
            raise ValueError('Selection requires development-only feedback')
        summary = find_completed(root, reuse_identity(spec, root))
        if summary is None:
            raise ValueError(f'Candidate missing completed result: {rel}. Resolve or document infeasibility before selecting.')
        if summary.get('timing_schema') != 'online_loop_v2':
            raise ValueError('Candidate timing schema mismatch')
        if not all(math.isfinite(float(summary[k])) for k in ('p_diag','s_initial','online_total_s')):
            raise ValueError('Non-finite candidate metrics')
        score = (float(summary['p_diag'])+float(summary['s_initial']))/2
        grouped.setdefault(spec['dataset']['name'], []).append((score, -float(summary['online_total_s']), rel, spec, summary))
    paths, evidence = [], []
    for ds, rows in grouped.items():
        winner = sorted(rows, key=lambda x:(-x[0], -x[1], x[2]))[0]
        _, _, rel, chosen, summary = winner
        evidence.append(dict(dataset=ds, selected_config=rel, selected_run=summary['run_id'],
                             score=winner[0], rule='max (P_diag+S_initial)/2; tie faster; then lexical config'))
        for seed in SEEDS:
            spec = base(ds, seed, 'er_static')
            spec.update(method_id='limited_search_static', experiment_id='L_selected', claim_ids=['L_selected'])
            spec['training'].update(new_batch=chosen['training']['new_batch'], replay_batch=chosen['training']['replay_batch'])
            spec['replay']['capacity'] = chosen['replay']['capacity']
            rel = f'configs/light24/selected/{ds}_s{seed}.yaml'
            write(root/rel, spec)
            paths.append(rel)
    write(root/'experiments/light24/selected.yaml', dict(study_id='light24_v1', name='selected',
          estimated_hours=ESTIMATES['selected'], configs=paths))
    out=root/'reports/light24';out.mkdir(parents=True,exist_ok=True)
    (out/'selection.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select', action='store_true')
    args=parser.parse_args()
    select_static() if args.select else emit()


if __name__ == '__main__':
    main()
