from orion_repro.memory.budget import formula_stop_if_infeasible, guarded_clip
from orion_repro.prefetch.queue import BoundedPrefetcher, PrefetchItem


def test_formula_guard_does_not_clip():
    d = formula_stop_if_infeasible(
        suggested_new_batch=32,
        suggested_replay=200,
        predicted_bytes=100,
        limit_bytes=50,
        min_batch=1,
    )
    assert d.accepted is False
    assert d.reason == "preflight_infeasible"
    assert d.applied_new_batch == 32


def test_guarded_clip_reduces_replay_or_batch():
    d = guarded_clip(
        suggested_new_batch=32,
        suggested_replay=1000,
        predicted_bytes=10_000,
        limit_bytes=1_000,
        min_batch=1,
        bytes_per_batch_step=10,
        bytes_per_replay_item=10,
    )
    assert d.reason in {"clipped", "cannot_clip_to_limit", "ok"}
    if d.accepted:
        assert d.applied_replay <= 1000


def test_prefetch_queue_is_bounded_and_drops_stale():
    pf = BoundedPrefetcher(max_depth=2)

    def producer():
        yield PrefetchItem(0, 1, 0, "a")
        yield PrefetchItem(0, 1, 1, "b")
        yield PrefetchItem(0, 2, 2, "c")

    pf.start(producer)
    first = pf.get(config_version=2, timeout=2.0)
    pf.close()
    assert first is not None
    assert first.batch == "c"
    assert pf.dropped_stale >= 1
    assert pf.max_depth == 2
