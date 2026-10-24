from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt


class EpochRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: PositiveInt
    train_loss: float
    validation_loss: float
    validation_fraud_mse: Optional[float] = None
    validation_fraud_recall: Optional[float] = None
    learning_rate: float
    duration_seconds: NonNegativeFloat


class TrainingHistory(BaseModel):
    records: List[EpochRecord] = Field(default_factory=list)
    best_epoch: Optional[PositiveInt] = None
    stopped_early: bool = False
    total_seconds: NonNegativeFloat = 0.0

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    def mark_best(self, epoch: int) -> None:
        self.best_epoch = epoch

    def finish(self, total_seconds: float, stopped_early: bool) -> None:
        self.total_seconds = total_seconds
        self.stopped_early = stopped_early

    @property
    def best_record(self) -> Optional[EpochRecord]:
        return next((record for record in self.records if record.epoch == self.best_epoch), None)
