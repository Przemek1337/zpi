from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import IsolationForest

from fraud_detection.data.transaction_set import TransactionSet


class IsolationForestBaseline:
    """Reference anomaly detector, trained like the autoencoder on legitimate transactions only."""

    def __init__(self, seed: int, n_estimators: int = 200, max_samples: int = 256) -> None:
        self._forest = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1
        )
        self.fit_seconds = 0.0
        self.predict_seconds = 0.0

    def fit(self, train: TransactionSet) -> IsolationForestBaseline:
        started = time.perf_counter()
        self._forest.fit(train.features)
        self.fit_seconds = time.perf_counter() - started
        return self

    def score(self, transactions: TransactionSet) -> np.ndarray:
        """Anomaly score, higher means more suspicious (same orientation as reconstruction error)."""
        started = time.perf_counter()
        scores = -self._forest.score_samples(transactions.features)
        self.predict_seconds = time.perf_counter() - started
        return scores
