"""Versioned reuse identities. Full provenance snapshots remain in every run."""
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from orion_repro.envcheck import parse_smi_memory_mib
from orion_repro.provenance import sha256_file, snapshot_source_tree


def _referenced_content_hashes(spec, root):
    refs = {}
    candidates = [
        spec.get('dataset', {}).get('split_manifest'),
        (spec.get('resource_envelope') or {}).get('schedule_path'),
        (spec.get('data_supply') or {}).get('profile_path'),
        spec.get('frozen_protocol_path'),
    ]
    for rel in candidates:
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            refs[str(rel)] = sha256_file(path)
    return refs


def actual_training_packages():
    versions = {}
    for name in ("torch", "torchvision"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", None)
        except Exception:
            versions[name] = None
    try:
        import avalanche
        versions["avalanche"] = avalanche.__version__
    except Exception:
        versions["avalanche"] = None
    try:
        import torch
        versions["torch_cuda"] = str(torch.version.cuda)
    except Exception:
        versions["torch_cuda"] = None
    return versions


def platform_fingerprint():
    gpus = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "uuid": parts[2],
                        "memory_mib": parse_smi_memory_mib(parts[3]),
                        "memory_mib_raw": parts[3],
                        "driver_version": parts[4],
                    }
                )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "gpus": gpus,
    }


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
    if spec.get('reuse_version') in {2, 3, 4}:
        files = sorted((root / 'src' / 'orion_repro').rglob('*.py'))
        files += [p for p in (root/'pyproject.toml',) if p.exists()]
        source = [(str(p.relative_to(root)), sha256_file(p)) for p in files]
    else:
        source = snapshot_source_tree(root)['aggregate_sha256']
    identity = {'config': clean, 'source': source,
                'environment': sha256_file(root / 'requirements.lock.txt'),
                'data_manifest': sha256_file(path)}
    if spec.get('reuse_version') == 3:
        identity['referenced_content'] = _referenced_content_hashes(spec, root)
        identity['resource_schedule'] = (spec.get('resource_envelope') or {}).get(
            'reserved_bytes_by_experience'
        )
        identity['data_supply'] = spec.get('data_supply')
    if spec.get('reuse_version') == 4:
        identity['schema'] = 'reuse_v4'
        identity['referenced_content'] = _referenced_content_hashes(spec, root)
        identity['resource_schedule'] = (spec.get('resource_envelope') or {}).get(
            'reserved_bytes_by_experience'
        )
        identity['data_supply'] = spec.get('data_supply')
        identity['environment_lock'] = identity['environment']
        identity['environment_actual'] = actual_training_packages()
        identity['platform'] = platform_fingerprint()
        identity['frozen_hash'] = spec.get('frozen_hash')
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def find_completed(root, identity):
    if identity is None:
        return None
    for path in sorted((root / 'runs').glob('*/summary.json'), reverse=True):
        summary = json.loads(path.read_text())
        if summary.get('status') == 'completed' and summary.get('reuse_identity') == identity:
            return summary
    return None
