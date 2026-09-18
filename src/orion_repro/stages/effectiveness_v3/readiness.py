"""Read-only entry check and explicit pre-freeze review contract."""
from __future__ import annotations
import json
from pathlib import Path
from orion_repro.stages.effectiveness_v3.context import StageContext, require_orion_interpreter
from orion_repro.stages.effectiveness_v3.identity import source_identity
from orion_repro.stages.effectiveness_v3.util import StageError, sha256_file

REVIEW_ITEMS = ('raw_evidence_identity', 'development_coverage', 'resource_scenarios',
                'paired_reporting', 'failure_and_integrity_tests')


def require_freeze_review(context: StageContext):
    path = context.revision_dir / 'g2_review.json'
    if not path.is_file():
        raise StageError('G2 review missing: complete SDD section 10 before freeze; write g2_review.json with evidence hashes')
    review = json.loads(path.read_text())
    identity = source_identity(context.root, design_path=context.design_path)
    if review.get('source_hash') != identity['source_hash'] or review.get('study_id') != context.study_id or review.get('revision') != context.revision:
        raise StageError('G2 review identity mismatch')
    for name in REVIEW_ITEMS:
        item = review.get('checks', {}).get(name, {})
        if item.get('status') != 'pass' or not item.get('artifacts'):
            raise StageError(f'G2 review incomplete: {name}')
        for rel, digest in item['artifacts'].items():
            artifact = (context.root / rel).resolve()
            if not artifact.is_relative_to(context.root) or not artifact.is_file() or sha256_file(artifact) != digest:
                raise StageError(f'G2 review artifact mismatch: {rel}')
    return review


def main():
    root = Path(__file__).resolve().parents[4]
    require_orion_interpreter()
    report = root / 'reports/effectiveness_v3/readiness/data.json'
    data = json.loads(report.read_text()) if report.is_file() else {}
    required = [root / 'data/raw/cifar100/cifar-100-python/train', root / 'data/raw/core50/core50_32x32',
                root / 'data/processed/core50_nc_run0_dev_split_seed17/val_filelist.txt']
    ready = bool(data.get('ready')) and all(p.exists() for p in required)
    for rel, digest in data.get('split_file_hashes', {}).items():
        p = root / rel
        ready = ready and p.is_file() and sha256_file(p) == digest
    print(json.dumps({'development_data_ready': ready, 'formal_ready': False,
                      'formal_reason': 'requires host G2 calibration, SDD section 10 review, then freeze',
                      'missing': [str(p.relative_to(root)) for p in required if not p.exists()]}, indent=2))
    raise SystemExit(0 if ready else 2)


if __name__ == '__main__':
    main()
