from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict
from torch import nn

from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.detection_metrics import detection_rate, threshold_for_fpr
from fraud_detection.evaluation.reconstruction_error_scorer import ReconstructionErrorScorer
from fraud_detection.exceptions import DataValidationError


class ReconstructionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    legitimate_mse: float
    fraud_mse: Optional[float] = None
    # share of fraud detected at the threshold keeping validation FPR below the configured budget
    fraud_recall: Optional[float] = None


class ReconstructionMonitor:
    def __init__(
        self,
        scorer: ReconstructionErrorScorer,
        validation: TransactionSet,
        legitimate_label: int,
        max_false_positive_rate: float = 0.01,
    ) -> None:
        self._scorer = scorer
        self._legitimate = validation.with_label(legitimate_label)
        self._fraud = validation.excluding_label(legitimate_label)
        self._max_false_positive_rate = max_false_positive_rate
        if len(self._legitimate) == 0:
            raise DataValidationError(
                f"Validation set contains no legitimate transactions (label {legitimate_label})"
            )

    def evaluate(self, model: nn.Module) -> ReconstructionSnapshot:
        legitimate_scores = self._scorer.score(model, self._legitimate.features)
        if len(self._fraud) == 0:
            return ReconstructionSnapshot(legitimate_mse=float(legitimate_scores.mean()))
        fraud_scores = self._scorer.score(model, self._fraud.features)
        threshold = threshold_for_fpr(legitimate_scores, self._max_false_positive_rate)
        return ReconstructionSnapshot(
            legitimate_mse=float(legitimate_scores.mean()),
            fraud_mse=float(fraud_scores.mean()),
            fraud_recall=detection_rate(fraud_scores, threshold),
        )
