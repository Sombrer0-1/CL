"""Versioned grouped holdout for classification patches (A22).

Numeric patch IDs are an ordering proxy, not verified camera timestamps.
Hold out one contiguous range per class and embargo neighboring IDs by rank.
"""
from pathlib import Path
import re
from collections import defaultdict
from orion_repro.rng import mix_seed

POLICY = "class_ordered_holdout_embargo_v2"


def grouped_endless_indices(paths, *, experience_id, seed=17, val_fraction=0.2, embargo=32):
    if not 0 < val_fraction < 1 or embargo < 1:
        raise ValueError("require 0 < val_fraction < 1 and embargo >= 1")
    groups = defaultdict(list)
    for i, path in enumerate(paths):
        p = Path(path)
        match = re.fullmatch(r"(.+)_(\d+)", p.stem)
        if not match:
            raise ValueError(f"cannot infer patch ordering: {p.name}")
        groups[str(p.parent)].append((int(match[2]), i))
    train, val, excluded = [], [], []
    for group, members in sorted(groups.items()):
        members.sort()
        if len({x[0] for x in members}) != len(members):
            raise ValueError(f"duplicate patch IDs in {group}")
        n = len(members)
        count = max(1, round(n * val_fraction))
        if n - count <= 2 * embargo:
            raise ValueError(f"class too short for holdout and embargo: {group}")
        # Root-independent, reproducible start; class name and experience identify group.
        start = mix_seed(seed, experience_id, Path(group).name, POLICY) % (n - count + 1)
        stop = start + count
        for rank, (_, index) in enumerate(members):
            if start <= rank < stop:
                val.append(index)
            elif max(0, start - embargo) <= rank < min(n, stop + embargo):
                excluded.append(index)
            else:
                train.append(index)
    return sorted(train), sorted(val), sorted(excluded)
