"""Plasticity / stability from an accuracy matrix (PLAN §5.1, A05).

Paper §2.1 defines P as mean acc_i and S via unspecified F_i.
This module implements the operational definitions in PLAN, plus aliases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOAT = np.float64


@dataclass
class ExperienceMetrics:
    k: int  # 1-based number of completed experiences
    p_diag: float
    s_initial: float
    avg_seen_accuracy: float
    avg_seen_accuracy_weighted: float | None
    forgetting_max: float
    stability_max: float
    current_experience_accuracy: float


def p_diag(diagonal: list[float]) -> float:
    if not diagonal:
        raise ValueError("diagonal must be non-empty")
    return float(np.mean(np.asarray(diagonal, dtype=FLOAT)))


def forgetting_initial(a_ii: float, a_ki: float) -> float:
    """F_initial = A[i,i] - A[k,i]; may be negative (negative forgetting)."""
    return float(FLOAT(a_ii) - FLOAT(a_ki))


def s_initial(forgetting_values: list[float], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return 1.0
    if len(forgetting_values) != k - 1:
        raise ValueError("need k-1 forgetting terms")
    return float(FLOAT(1.0) - np.mean(np.asarray(forgetting_values, dtype=FLOAT)))


def summarize_matrix(
    matrix: np.ndarray,
    totals: np.ndarray | None = None,
) -> ExperienceMetrics:
    """matrix[k0, i0] = accuracy after training experience k0+1 on domain i0+1.

    Only the leading k x k block is used, where k is the last filled row+1.
    Unfilled entries should be NaN.
    """
    mat = np.asarray(matrix, dtype=FLOAT)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("accuracy matrix must be square")
    filled = [r for r in range(mat.shape[0]) if np.isfinite(mat[r, r])]
    if not filled:
        raise ValueError("no completed experiences")
    k = filled[-1] + 1
    block = mat[:k, :k]
    diag = [float(block[i, i]) for i in range(k)]
    p = p_diag(diag)
    if k == 1:
        s = 1.0
        fmax = 0.0
    else:
        f_init = [forgetting_initial(block[i, i], block[k - 1, i]) for i in range(k - 1)]
        s = s_initial(f_init, k)
        peaks = []
        for i in range(k - 1):
            hist = block[i:k, i]
            peaks.append(float(np.max(hist) - block[k - 1, i]))
        fmax = float(np.mean(np.asarray(peaks, dtype=FLOAT)))
    seen = block[k - 1, :k]
    avg_seen = float(np.mean(seen))
    weighted = None
    if totals is not None:
        tot = np.asarray(totals, dtype=FLOAT)[:k]
        if np.all(tot > 0) and np.all(np.isfinite(tot)):
            weighted = float(np.sum(seen * tot) / np.sum(tot))
    return ExperienceMetrics(
        k=k,
        p_diag=p,
        s_initial=s,
        avg_seen_accuracy=avg_seen,
        avg_seen_accuracy_weighted=weighted,
        forgetting_max=fmax,
        stability_max=float(FLOAT(1.0) - FLOAT(fmax)) if k > 1 else 1.0,
        current_experience_accuracy=float(block[k - 1, k - 1]),
    )
