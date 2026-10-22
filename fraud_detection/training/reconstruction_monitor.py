from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict
from torch import nn

from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.reconstruction_error_scorer import ReconstructionErrorScorer
from fraud_detection.exceptions import DataValidationError


class ReconstructionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    legitimate_mse: float
    fraud_mse: Optional[float] = None


class ReconstructionMonitor:
    def __init__(self, scorer: ReconstructionErrorScorer, validation: TransactionSet, legitimate_label: int) -> None:
        self._scorer = scorer
        self._legitimate = validation.with_label(legitimate_label)
        self._fraud = validation.excluding_label(legitimate_label)
        if len(self._legitimate) == 0:
            raise DataValidationError(
                f"Validation set contains no legitimate transactions (label {legitimate_label})"
            )

    def evaluate(self, model: nn.Module) -> ReconstructionSnapshot:
        return ReconstructionSnapshot(
            legitimate_mse=self._mean_error(model, self._legitimate),
            fraud_mse=self._mean_error(model, self._fraud) if len(self._fraud) > 0 else None,
        )

    def _mean_error(self, model: nn.Module, transactions: TransactionSet) -> float:
        return float(self._scorer.score(model, transactions.features).mean())
