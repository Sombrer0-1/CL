"""Report current-study coverage by full identity; preserve all attempts separately."""
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from orion_repro.runner.spec import load_yaml
from orion_repro.runner.reuse import reuse_identity

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    out=ROOT/'reports/light24';out.mkdir(parents=True,exist_ok=True)
    registry={}
    for line in (ROOT/'experiments/registry.jsonl').read_text().splitlines():
        event=json.loads(line);registry[event['run_id']]={**registry.get(event['run_id'],{}),**event}
    ledger_path=ROOT/'runs/light24_progress.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'results':[]}
    ledger_rows={a['run_id']:a for a in ledger['results']}
    if ledger.get('active'):
        ledger_rows[ledger['active']['run_id']]={**ledger['active'], 'status':'incomplete'}
    attempts=[]
    for path in sorted((ROOT/'runs').glob('*/resolved_config.yaml.json')):
        spec=json.loads(path.read_text())
        if spec.get('study_id') != 'light24_v1':continue
        summary_path=path.parent/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary={**summary, **{k:v for k,v in registry.get(path.parent.name,{}).items() if k=='status'}}
        attempts.append(dict(run_id=path.parent.name, identity=summary.get('reuse_identity') or ledger_rows.get(path.parent.name,{}).get('identity'),
            status=summary.get('status','incomplete'), experiment=spec['experiment_id'],
            dataset=spec['dataset']['name'], method=spec['method_id'],seed=spec['seeds']['model'],
            P=summary.get('p_diag'), S=summary.get('s_initial'), online_s=summary.get('online_total_s')))
    seen={a['run_id'] for a in attempts}
    for rid, row in ledger_rows.items():
        if rid in seen:continue
        spec=load_yaml(ROOT/row['config'])
        attempts.append(dict(run_id=rid, identity=row['identity'], status=row['status'],
            experiment=spec['experiment_id'], dataset=spec['dataset']['name'], method=spec['method_id'],
            seed=spec['seeds']['model'], P=None, S=None, online_s=None))
    expected=load_yaml(ROOT/'experiments/light24/all.yaml')['configs']
    selected=ROOT/'experiments/light24/selected.yaml'
    if selected.exists():expected+=load_yaml(selected)['configs']
    rows=[]
    for rel in expected:
        spec=load_yaml(ROOT/rel);identity=reuse_identity(spec,ROOT)
        hits=[a for a in attempts if a['identity']==identity]
        completed=[a for a in hits if a['status']=='completed']
        hit=(completed or hits or [{}])[-1]
        variant=Path(rel).stem.rsplit('_s',1)[0] if spec['phase']=='formal' else Path(rel).stem
        rows.append(dict(config=rel, experiment=spec['experiment_id'], dataset=spec['dataset']['name'],
                         variant=variant, seed=spec['seeds']['model'],status=hit.get('status','not_run'),
                         run_id=hit.get('run_id'), P=hit.get('P'), S=hit.get('S'),online_s=hit.get('online_s')))
    write_csv(out/'coverage.csv',rows,list(rows[0]))
    write_csv(out/'attempts.csv', attempts, ['run_id','identity','status','experiment','dataset','method','seed','P','S','online_s'])
    groups=defaultdict(list)
    for row in rows:groups[(row['experiment'],row['variant'])].append(row)
    means=[]
    for (experiment,variant), cells in groups.items():
        valid=[c for c in cells if c['status']=='completed' and all(c[k] is not None for k in ('P','S','online_s'))]
        item=dict(experiment=experiment,variant=variant,n_expected=len(cells),n_completed=len(valid))
        for metric in ('P','S','online_s'):
            vals=[float(c[metric]) for c in valid]
            item[metric+'_mean']=mean(vals) if vals else None
            item[metric+'_sd']=stdev(vals) if len(vals)>1 else None
        means.append(item)
    write_csv(out/'means.csv',means,list(means[0]))
    payload=dict(n_emitted=len(rows),n_planned=96,n_completed=sum(r['status']=='completed' for r in rows),
                 selected_configs_ready=selected.exists(),note='Current identity only; missing and failed results are not zeros. Historical runs excluded. Static selection adds 6 cells after 12 development candidates.')
    (out/'progress.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
