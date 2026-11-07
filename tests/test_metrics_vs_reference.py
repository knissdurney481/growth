import numpy as np

from growth_twin.appraisal.metrics import mrr, recall_at_k


def test_mrr_and_recall_closed_form() -> None:
    scores = np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]])
    labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(mrr(scores, labels) - (1.0 + 0.5) / 2.0) < 1e-9
    assert abs(recall_at_k(scores, labels, 1) - 0.5) < 1e-9
