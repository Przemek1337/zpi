from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

import optuna
import torch

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.exceptions import TrainingError
from fraud_detection.training.training_history import EpochRecord
from fraud_detection.training.training_runner import TrainingRunner
from fraud_detection.tuning.search_space import SearchSpace, apply_overrides
from fraud_detection.tuning.tuning_config import TuningConfig

TRAINING_LOGGER = "fraud_detection.training"


def trial_config(base: ExperimentConfig, tuning: TuningConfig, trial: optuna.trial.BaseTrial) -> ExperimentConfig:
    """Experiment config for one search trial: sampled hyperparameters plus the per-trial epoch budget."""
    return apply_overrides(
        base,
        SearchSpace(tuning.search_space).suggest(trial),
        {
            "training": {"epochs": tuning.trial.epochs},
            "early_stopping": {"enabled": True, "patience": tuning.trial.early_stopping_patience},
        },
    )


def final_config(base: ExperimentConfig, tuning: TuningConfig, best: optuna.trial.FrozenTrial) -> ExperimentConfig:
    """Experiment config for retraining the winner with the base config's full epoch budget."""
    return apply_overrides(
        base,
        SearchSpace(tuning.search_space).suggest(optuna.trial.FixedTrial(best.params)),
        {"output": {"directory": str(tuning.output.directory / "final_model")}},
    )


class TrialObjective:
    """Optuna objective: validation fraud recall at the configured false positive budget.

    The score is taken at the epoch with the lowest validation reconstruction loss, which is the
    checkpoint a regular training run would keep.
    """

    def __init__(
        self,
        base: ExperimentConfig,
        tuning: TuningConfig,
        train: TransactionSet,
        validation: TransactionSet,
        device: torch.device,
    ) -> None:
        self._base = base
        self._tuning = tuning
        self._train = train
        self._validation = validation
        self._device = device

    def __call__(self, trial: optuna.trial.Trial) -> float:
        config = trial_config(self._base, self._tuning, trial)
        trial.set_user_attr("hidden_dims", list(config.model.hidden_dims))
        started = time.perf_counter()
        with quiet_training_logs():
            model, history = TrainingRunner(config).train(
                self._train, self._validation, self._device, self._pruning_callback(trial)
            )
        best = history.best_record
        if best is None or best.validation_fraud_recall is None:
            raise TrainingError("Trial produced no validation fraud recall")
        trial.set_user_attr("parameters", model.parameter_count)
        trial.set_user_attr("best_epoch", best.epoch)
        trial.set_user_attr("epochs_run", len(history.records))
        trial.set_user_attr("val_legit_mse", best.validation_loss)
        trial.set_user_attr("val_fraud_mse", best.validation_fraud_mse)
        trial.set_user_attr("minutes", round((time.perf_counter() - started) / 60, 2))
        return best.validation_fraud_recall

    def _pruning_callback(self, trial: optuna.trial.Trial):
        def report(record: EpochRecord) -> None:
            if record.validation_fraud_recall is None:
                return
            trial.report(record.validation_fraud_recall, record.epoch)
            if self._tuning.pruning.enabled and trial.should_prune():
                trial.set_user_attr("pruned_at_epoch", record.epoch)
                raise optuna.TrialPruned(f"pruned at epoch {record.epoch}")

        return report


@contextmanager
def quiet_training_logs() -> Iterator[None]:
    """Hides per-epoch training logs; trials are summarised one line each instead."""
    training_logger = logging.getLogger(TRAINING_LOGGER)
    previous = training_logger.level
    training_logger.setLevel(max(logging.WARNING, logging.getLogger().getEffectiveLevel()))
    try:
        yield
    finally:
        training_logger.setLevel(previous)
