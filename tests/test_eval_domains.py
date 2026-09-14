import numpy as np

from orion_repro.evaluation.domains import class_sets_disjoint, eval_protocol
from orion_repro.evaluation.metrics import summarize_matrix


def test_class_sets_disjoint_detects_overlap():
    assert class_sets_disjoint([[0, 1], [2, 3]]) is True
    assert class_sets_disjoint([[0, 1], [1, 2]]) is False


class _Stream:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _Bench:
    def __init__(self, n_test):
        self.test_stream = _Stream(n_test)


def test_eval_protocol_names():
    cmap = [[0, 1], [2, 3]]
    assert eval_protocol(_Bench(10), cmap) == "per_experience"
    assert eval_protocol(_Bench(1), cmap) == "shared_sliced_by_train_classes"
    assert eval_protocol(_Bench(1), [[0, 1], [0, 2]]) == "shared_test_temporal"


def test_shared_test_temporal_ps_is_history_of_a_k():
    """PLAN §5.3: A[k, i] = a_k. P_diag is mean a_i; S is not class-wise forgetting."""
    a = [0.40, 0.50, 0.45]
    mat = np.full((3, 3), np.nan)
    for k, ak in enumerate(a):
        mat[k, : k + 1] = ak
    m = summarize_matrix(mat)
    assert abs(m.p_diag - float(np.mean(a))) < 1e-12
    f_init = [a[0] - a[2], a[1] - a[2]]
    assert abs(m.s_initial - (1.0 - float(np.mean(f_init)))) < 1e-12
    assert abs(m.current_experience_accuracy - a[-1]) < 1e-12
