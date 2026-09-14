from pathlib import Path
import pytest
from orion_repro.benchmarks.endless_split import grouped_endless_indices


def test_grouped_split_uses_patch_ids_not_filesystem_order():
    paths = [f'/root/0/{c}/{c}_{i}.png' for c in ('Car', 'Tree') for i in reversed(range(200))]
    tr, va, excluded = grouped_endless_indices(paths, experience_id=0, embargo=8)
    assert len(va) == 80
    assert sorted(tr + va + excluded) == list(range(400))
    assert not set(tr) & set(va)
    for c in ('Car', 'Tree'):
        tid = {int(Path(paths[i]).stem.rsplit('_', 1)[1]) for i in tr if Path(paths[i]).parent.name == c}
        vid = {int(Path(paths[i]).stem.rsplit('_', 1)[1]) for i in va if Path(paths[i]).parent.name == c}
        assert max(vid) - min(vid) + 1 == len(vid)
        assert min(abs(t-v) for t in tid for v in vid) > 8
    shuffled = list(reversed(paths))
    tr2, va2, ex2 = grouped_endless_indices(shuffled, experience_id=0, embargo=8)
    assert {paths[i] for i in tr} == {shuffled[i] for i in tr2}
    assert {paths[i] for i in va} == {shuffled[i] for i in va2}


def test_ambiguous_or_too_small_groups_fail():
    with pytest.raises(ValueError):
        grouped_endless_indices(['/a/unknown.png'], experience_id=0)
    with pytest.raises(ValueError):
        grouped_endless_indices([f'/a/a_{i}.png' for i in range(10)], experience_id=0)
