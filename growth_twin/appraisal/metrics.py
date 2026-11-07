import numpy as np
from numpy.typing import NDArray


def _np(x: object) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def mad(pred: object, target: object) -> float:
    return float(np.mean(np.abs(_np(pred) - _np(target))))


def rmse(pred: object, target: object) -> float:
    return float(np.sqrt(np.mean((_np(pred) - _np(target)) ** 2)))


def pearson_r(pred: object, target: object) -> float:
    a = _np(pred)
    b = _np(target)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def within_months(pred: object, target: object, months: float) -> float:
    return float(np.mean(np.abs(_np(pred) - _np(target)) <= months))


def auroc(scores: object, labels: object) -> float:
    s = _np(scores)
    y = _np(labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: object, labels: object) -> float:
    s = _np(scores)
    y = _np(labels)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    total_pos = y.sum()
    if total_pos == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def precision_at_k(scores: NDArray[np.float64], labels: NDArray[np.float64], k: int) -> float:
    out = []
    for row_s, row_y in zip(scores, labels, strict=True):
        top = np.argsort(-row_s, kind="mergesort")[:k]
        out.append(row_y[top].sum() / k)
    return float(np.mean(out))


def recall_at_k(scores: NDArray[np.float64], labels: NDArray[np.float64], k: int) -> float:
    out = []
    for row_s, row_y in zip(scores, labels, strict=True):
        total = row_y.sum()
        if total == 0:
            continue
        top = np.argsort(-row_s, kind="mergesort")[:k]
        out.append(row_y[top].sum() / total)
    return float(np.mean(out)) if out else 0.0


def mrr(scores: NDArray[np.float64], labels: NDArray[np.float64]) -> float:
    out = []
    for row_s, row_y in zip(scores, labels, strict=True):
        order = np.argsort(-row_s, kind="mergesort")
        ranked = row_y[order]
        hits = np.where(ranked > 0.5)[0]
        if len(hits) == 0:
            continue
        out.append(1.0 / (hits[0] + 1))
    return float(np.mean(out)) if out else 0.0
