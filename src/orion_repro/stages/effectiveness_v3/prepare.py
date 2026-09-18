"""Restore public v3 data without rewriting historical manifests."""
from __future__ import annotations

import json
from pathlib import Path

from orion_repro.prepare_data import prepare_cifar100
from orion_repro.prepare_core50 import prepare_core50_mini
from orion_repro.benchmarks.dev_split import build_cifar_dev_manifest
from orion_repro.benchmarks.core50_split import write_core50_dev_split
from orion_repro.stages.effectiveness_v3.context import require_orion_interpreter
from orion_repro.stages.effectiveness_v3.util import sha256_file

ROOT = Path(__file__).resolve().parents[4]


def main():
    require_orion_interpreter()
    manifests = ROOT / 'data/manifests'
    before = {str(p): sha256_file(p) for p in manifests.rglob('*') if p.is_file()}
    out = ROOT / 'reports/effectiveness_v3/readiness'
    out.mkdir(parents=True, exist_ok=True)
    cifar = prepare_cifar100(ROOT / 'data/raw/cifar100')
    from torchvision.datasets import CIFAR100
    train = CIFAR100(str(ROOT / 'data/raw/cifar100'), train=True, download=False)
    split = build_cifar_dev_manifest(dataset='cifar100', labels=list(train.targets))
    old = json.loads((manifests / 'cifar100_dev_split_seed17.json').read_text())
    for key in ('train_indices', 'val_indices', 'split_seed'):
        if split[key] != old[key]:
            raise ValueError(f'CIFAR historical split mismatch: {key}')
    (out / 'cifar100.json').write_text(json.dumps(cifar, indent=2))
    core = prepare_core50_mini(ROOT / 'data/raw/core50')
    split = write_core50_dev_split(dataset_root=ROOT / 'data/raw/core50', scenario='nc')
    old = json.loads((manifests / 'core50_nc_run0_dev_split_seed17.json').read_text())
    for key in ('train_sizes', 'val_sizes_per_experience', 'n_dev_train', 'n_dev_val', 'split_seed'):
        if split[key] != old[key]:
            raise ValueError(f'CORe50 historical split mismatch: {key}')
    files = sorted((ROOT / 'data/processed/core50_nc_run0_dev_split_seed17').glob('*.txt'))
    # Validate every referenced image, rather than treating directory existence as readiness.
    image_root = ROOT / 'data/raw/core50/core50_32x32'
    official_test = ROOT / 'data/raw/core50/batches_filelists/NC_inc/run0/test_filelist.txt'
    for file in [*files, official_test]:
        for line in file.read_text().splitlines():
            if line.strip() and not (image_root / line.split()[0]).is_file():
                raise ValueError(f'missing image referenced by {file}: {line}')
    after = {str(p): sha256_file(p) for p in manifests.rglob('*') if p.is_file()}
    if before != after:
        raise RuntimeError('historical manifests changed during preparation')
    payload = {'ready': True, 'cifar100': cifar, 'core50': core, 'core50_split': split,
               'split_file_hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
               'historical_manifests_unchanged': True,
               'archive_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (ROOT / 'data/raw/core50').glob('*.zip')}}
    (out / 'data.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'ready': True, 'report': str(out / 'data.json')}))


if __name__ == '__main__':
    main()
