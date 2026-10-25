from __future__ import annotations

import math

from fraud_detection.configuration.experiment_config import EarlyStoppingConfig


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, enabled: bool = True) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._enabled = enabled
        self._best_loss = math.inf
        self._epochs_without_improvement = 0

    @classmethod
    def from_config(cls, config: EarlyStoppingConfig) -> EarlyStopping:
        return cls(patience=config.patience, min_delta=config.min_delta, enabled=config.enabled)

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def should_stop(self) -> bool:
        return self._enabled and self._epochs_without_improvement >= self._patience

    def update(self, loss: float) -> bool:
        if loss < self._best_loss - self._min_delta:
            self._best_loss = loss
            self._epochs_without_improvement = 0
            return True
        self._epochs_without_improvement += 1
        return False
