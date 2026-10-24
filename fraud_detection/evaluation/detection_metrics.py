from __future__ import annotations

import math

import numpy as np


def threshold_for_fpr(legitimate_scores: np.ndarray, max_fpr: float) -> float:
    """Lowest threshold whose false positive rate on ``legitimate_scores`` stays strictly below ``max_fpr``.

    A transaction is flagged as fraud when its score is strictly greater than the threshold.
    """
    if len(legitimate_scores) == 0:
        raise ValueError("cannot derive a threshold without legitimate scores")
    if not 0.0 < max_fpr < 1.0:
        raise ValueError(f"max_fpr must lie in (0, 1), got {max_fpr}")
    ordered = np.sort(legitimate_scores)
    allowed_false_positives = math.ceil(max_fpr * len(ordered)) - 1
    return float(ordered[len(ordered) - 1 - allowed_false_positives])


def detection_rate(fraud_scores: np.ndarray, threshold: float) -> float:
    if len(fraud_scores) == 0:
        raise ValueError("cannot compute detection rate without fraud scores")
    return float(np.mean(fraud_scores > threshold))
