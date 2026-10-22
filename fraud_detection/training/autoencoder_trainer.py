from __future__ import annotations

import copy
import math
import time
from typing import Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from fraud_detection.configuration.experiment_config import TrainingConfig
from fraud_detection.exceptions import TrainingError
from fraud_detection.training.early_stopping import EarlyStopping
from fraud_detection.training.reconstruction_monitor import ReconstructionMonitor
from fraud_detection.training.training_history import EpochRecord, TrainingHistory
from fraud_detection.utils.colored_logging import get_logger

logger = get_logger(__name__)


class AutoencoderTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        early_stopping: EarlyStopping,
        config: TrainingConfig,
        device: torch.device,
        scheduler: Optional[ReduceLROnPlateau] = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._early_stopping = early_stopping
        self._config = config
        self._device = device
        self._scheduler = scheduler
        self._loss_fn = nn.MSELoss()

    def fit(self, train_loader: DataLoader, monitor: ReconstructionMonitor) -> TrainingHistory:
        history = TrainingHistory()
        started = time.perf_counter()
        try:
            best_state = self._run_epochs(train_loader, monitor, history)
        except torch.cuda.OutOfMemoryError as error:
            raise TrainingError(
                f"GPU ran out of memory; reduce training.batch_size (currently {self._config.batch_size})"
            ) from error
        self._model.load_state_dict(best_state)
        history.finish(time.perf_counter() - started, self._early_stopping.should_stop)
        logger.info("Training finished in %.1fs, best epoch %s", history.total_seconds, history.best_epoch)
        return history

    def _run_epochs(self, train_loader: DataLoader, monitor: ReconstructionMonitor, history: TrainingHistory) -> dict:
        best_state = copy.deepcopy(self._model.state_dict())
        for epoch in range(1, self._config.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = self._train_epoch(train_loader, epoch)
            snapshot = monitor.evaluate(self._model)
            self._ensure_finite(snapshot.legitimate_mse, f"validation loss at epoch {epoch}")
            self._step_scheduler(snapshot.legitimate_mse)
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=snapshot.legitimate_mse,
                validation_fraud_mse=snapshot.fraud_mse,
                learning_rate=self._current_learning_rate(),
                duration_seconds=time.perf_counter() - epoch_started,
            )
            history.append(record)
            self._log(record)

            if self._early_stopping.update(snapshot.legitimate_mse):
                best_state = copy.deepcopy(self._model.state_dict())
                history.mark_best(epoch)
                logger.debug("New best validation loss %.6f at epoch %d", snapshot.legitimate_mse, epoch)
            if self._early_stopping.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        return best_state

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self._model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_index, (batch,) in enumerate(loader, start=1):
            batch = batch.to(self._device, non_blocking=True)
            self._optimizer.zero_grad(set_to_none=True)
            loss = self._loss_fn(self._model(batch), batch)
            self._ensure_finite(loss.item(), f"training loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            self._clip_gradients()
            self._optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)
        if total_samples == 0:
            raise TrainingError("Training data loader produced no batches")
        return total_loss / total_samples

    @staticmethod
    def _ensure_finite(value: float, description: str) -> None:
        if not math.isfinite(value):
            raise TrainingError(
                f"Non-finite {description} ({value}); lower the learning rate or verify feature scaling"
            )

    def _clip_gradients(self) -> None:
        if self._config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self._model.parameters(), self._config.gradient_clip_norm)

    def _step_scheduler(self, validation_loss: float) -> None:
        if self._scheduler is not None:
            self._scheduler.step(validation_loss)

    def _current_learning_rate(self) -> float:
        return self._optimizer.param_groups[0]["lr"]

    def _log(self, record: EpochRecord) -> None:
        fraud_mse = "n/a" if record.validation_fraud_mse is None else f"{record.validation_fraud_mse:.6f}"
        logger.info(
            "Epoch %d/%d | train %.6f | val legit %.6f | val fraud %s | lr %.2e | %.1fs",
            record.epoch,
            self._config.epochs,
            record.train_loss,
            record.validation_loss,
            fraud_mse,
            record.learning_rate,
            record.duration_seconds,
        )
