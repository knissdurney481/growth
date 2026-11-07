import itertools
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def bootstrap_ci(
    sample: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    reps: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(sample)
    draws = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        draws[i] = statistic(sample[idx])
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lo, hi


def paired_t_test(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[float, float]:
    result = stats.ttest_rel(a, b)
    return float(result.statistic), float(result.pvalue)


def bonferroni(p_value: float, comparisons: int) -> float:
    return float(min(1.0, p_value * comparisons))


def _midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(
    labels: NDArray[np.float64], preds_a: NDArray[np.float64], preds_b: NDArray[np.float64]
) -> tuple[float, float, float]:
    order = np.argsort(-labels, kind="mergesort")
    y = labels[order]
    m = int((y > 0.5).sum())
    stacked = np.vstack([preds_a[order], preds_b[order]])
    n = stacked.shape[1] - m
    positive = stacked[:, :m]
    negative = stacked[:, m:]
    k = stacked.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(stacked[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def expected_calibration_error(
    confidence: NDArray[np.float64], correct: NDArray[np.float64], n_bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in itertools.pairwise(edges):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def coverage(
    lower: NDArray[np.float64], upper: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    inside = (target >= lower) & (target <= upper)
    return float(np.mean(inside))
