from orion_repro.prefetch.dataloader import PrefetchingDataLoader
from orion_repro.prefetch.queue import BoundedPrefetcher, PrefetchItem


class _SeqLoader:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def test_prefetch_preserves_inner_order():
    items = list(range(32))
    wrapped = PrefetchingDataLoader(
        _SeqLoader(items), experience_id=3, config_version=1, max_depth=2
    )
    assert list(wrapped) == items
    assert wrapped.batches_consumed == 32


def test_prefetch_early_close_does_not_deadlock():
    wrapped = PrefetchingDataLoader(
        _SeqLoader(list(range(200))), experience_id=0, config_version=0, max_depth=2
    )
    iterator = iter(wrapped)
    assert next(iterator) == 0
    assert next(iterator) == 1
    iterator.close()


def test_stale_config_is_dropped_and_counted():
    pf = BoundedPrefetcher(max_depth=4)

    def producer():
        yield PrefetchItem(0, 1, 0, "old")
        yield PrefetchItem(0, 2, 1, "new")

    pf.start(producer)
    item = pf.get(config_version=2, timeout=2.0)
    pf.close()
    assert item is not None
    assert item.batch == "new"
    assert pf.dropped_stale >= 1
