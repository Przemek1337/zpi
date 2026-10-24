import numpy as np
import pytest

from fraud_detection.evaluation.detection_metrics import detection_rate, threshold_for_fpr


def false_positive_rate(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores > threshold))


@pytest.mark.parametrize("count", [1, 99, 100, 101, 250, 42488])
def test_threshold_keeps_fpr_strictly_below_budget(count):
    scores = np.random.default_rng(count).normal(size=count)

    threshold = threshold_for_fpr(scores, 0.01)

    assert false_positive_rate(scores, threshold) < 0.01


def test_threshold_is_the_lowest_one_within_budget():
    scores = np.arange(1000, dtype=np.float64)

    threshold = threshold_for_fpr(scores, 0.01)

    assert threshold == 990.0
    assert false_positive_rate(scores, threshold) == pytest.approx(0.009)
    assert false_positive_rate(scores, threshold - 1) == pytest.approx(0.01)


def test_tied_scores_never_exceed_budget():
    scores = np.array([0.0] * 50 + [1.0] * 50)

    assert false_positive_rate(scores, threshold_for_fpr(scores, 0.3)) < 0.3


@pytest.mark.parametrize("max_fpr", [0.0, 1.0, -0.1])
def test_invalid_budget_is_rejected(max_fpr):
    with pytest.raises(ValueError, match="max_fpr"):
        threshold_for_fpr(np.ones(10), max_fpr)


def test_empty_scores_are_rejected():
    with pytest.raises(ValueError, match="legitimate"):
        threshold_for_fpr(np.empty(0), 0.01)
    with pytest.raises(ValueError, match="fraud"):
        detection_rate(np.empty(0), 0.5)


def test_detection_rate_counts_scores_strictly_above_threshold():
    assert detection_rate(np.array([0.1, 0.5, 0.9, 2.0]), 0.5) == 0.5
