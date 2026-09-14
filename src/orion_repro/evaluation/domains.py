"""Build per-experience eval datasets, including CORe50 shared-test (A14)."""

from __future__ import annotations

from typing import Any


def class_sets_disjoint(class_map: list[list[int]]) -> bool:
    seen: set[int] = set()
    for classes in class_map:
        current = {int(c) for c in classes}
        if current & seen:
            return False
        seen |= current
    return True


def eval_protocol(benchmark, class_map: list[list[int]]) -> str:
    if len(benchmark.test_stream) > 1:
        return "per_experience"
    if class_sets_disjoint(class_map):
        return "shared_sliced_by_train_classes"
    return "shared_test_temporal"


def subset_by_labels(dataset, classes: list[int]):
    wanted = {int(c) for c in classes}
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError("dataset has no .targets; cannot slice shared test by class")
    indices = [i for i, y in enumerate(targets) if int(y) in wanted]
    if hasattr(dataset, "subset"):
        return dataset.subset(indices)
    raise TypeError(f"cannot subset {type(dataset)}")


def make_eval_domains(benchmark, class_map: list[list[int]], k: int) -> list[tuple[int, Any]]:
    protocol = eval_protocol(benchmark, class_map)
    if protocol == "per_experience":
        return [(i, benchmark.test_stream[i].dataset) for i in range(k + 1)]
    shared = benchmark.test_stream[0].dataset
    if protocol == "shared_sliced_by_train_classes":
        return [(i, subset_by_labels(shared, class_map[i])) for i in range(k + 1)]
    return [(0, shared)]
