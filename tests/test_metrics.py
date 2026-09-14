import numpy as np

from orion_repro.evaluation.metrics import summarize_matrix


def test_hand_example_plan_section_13():
    mat = np.full((3, 3), np.nan)
    mat[0, 0] = 0.8
    mat[1, 0] = 0.6
    mat[1, 1] = 0.7
    mat[2, 0] = 0.7
    mat[2, 1] = 0.65
    mat[2, 2] = 0.9
    m = summarize_matrix(mat)
    assert m.k == 3
    assert abs(m.p_diag - 0.8) < 1e-12
    assert abs(m.s_initial - 0.925) < 1e-12
    assert abs(m.avg_seen_accuracy - 0.75) < 1e-12


def test_negative_forgetting_not_clipped():
    mat = np.full((2, 2), np.nan)
    mat[0, 0] = 0.6
    mat[1, 0] = 0.8
    mat[1, 1] = 0.7
    m = summarize_matrix(mat)
    assert abs(m.s_initial - 1.2) < 1e-12
    assert abs(m.forgetting_max - 0.0) < 1e-12
    assert abs(m.stability_max - 1.0) < 1e-12
