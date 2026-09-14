"""Versioned reuse identities. Full provenance snapshots remain in every run."""
import copy
import hashlib
import json
from pathlib import Path
from orion_repro.provenance import sha256_file, snapshot_source_tree


def reuse_identity(spec, root):
    manifest = spec.get('dataset', {}).get('split_manifest')
    if not manifest:
        return None
    path = Path(manifest)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return None
    clean = copy.deepcopy(spec)
    for key in ('run_id', 'code_revision_or_snapshot', 'environment_lock_sha256',
                'dataset_manifest_sha256', 'experiment_id', 'claim_ids'):
        clean.pop(key, None)
    # The original v1 identity is retained for configs not opting into v2.
    if spec.get('reuse_version') == 2:
        files = sorted((root / 'src' / 'orion_repro').rglob('*.py'))
        files += [p for p in (root/'pyproject.toml',) if p.exists()]
        source = [(str(p.relative_to(root)), sha256_file(p)) for p in files]
    else:
        source = snapshot_source_tree(root)['aggregate_sha256']
    identity = {'config': clean, 'source': source,
                'environment': sha256_file(root / 'requirements.lock.txt'),
                'data_manifest': sha256_file(path)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def find_completed(root, identity):
    if identity is None:
        return None
    for path in sorted((root / 'runs').glob('*/summary.json'), reverse=True):
        summary = json.loads(path.read_text())
        if summary.get('status') == 'completed' and summary.get('reuse_identity') == identity:
            return summary
    return None
