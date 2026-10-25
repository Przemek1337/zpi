from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.data.dataset_source import load_train_validation
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.reconstruction_error_scorer import ReconstructionErrorScorer
from fraud_detection.exceptions import DataValidationError
from fraud_detection.model.deep_autoencoder import DeepAutoencoder
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.training.autoencoder_trainer import AutoencoderTrainer, EpochCallback
from fraud_detection.training.early_stopping import EarlyStopping
from fraud_detection.training.optimizer_factory import create_optimizer, create_scheduler
from fraud_detection.training.reconstruction_monitor import ReconstructionMonitor
from fraud_detection.training.training_history import TrainingHistory
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.device_resolver import resolve_device
from fraud_detection.utils.reproducibility import seed_everything

logger = get_logger(__name__)


class TrainingRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._artifacts = ModelArtifactStore(config.output.directory)

    def run(self) -> TrainingHistory:
        device = resolve_device(self._config.device)
        logger.info("Using device: %s", device)
        train, validation = self.load_datasets()
        model, history = self.train(train, validation, device)
        self._log_summary(history)
        self._save_artifacts(model, train.feature_names, history)
        return history

    def train(
        self,
        train: TransactionSet,
        validation: TransactionSet,
        device: torch.device,
        epoch_callback: Optional[EpochCallback] = None,
    ) -> Tuple[DeepAutoencoder, TrainingHistory]:
        """Trains a fresh model on already loaded data without persisting anything."""
        seed_everything(self._config.seed)
        model = DeepAutoencoder(train.feature_count, self._config.model).to(device)
        logger.info("Model built with %d trainable parameters", model.parameter_count)
        logger.debug("Model architecture:\n%s", model)
        history = self._build_trainer(model, device).fit(
            self._train_loader(train, device),
            self._build_monitor(validation, device),
            epoch_callback,
        )
        return model, history

    def load_datasets(self) -> Tuple[TransactionSet, TransactionSet]:
        """Returns (legitimate-only training set, full validation set)."""
        data_config = self._config.data
        loaded_train, validation = load_train_validation(data_config)
        train = loaded_train.with_label(data_config.legitimate_label)
        if len(train) < len(loaded_train):
            logger.warning(
                "Removed %d non-legitimate transactions from training set", len(loaded_train) - len(train)
            )
        if len(train) == 0:
            source = data_config.splits_path or data_config.train_path
            raise DataValidationError(
                f"{source}: no legitimate transactions (label {data_config.legitimate_label})"
            )
        self._ensure_matching_features(train, validation)
        logger.info("Training set: %d legitimate transactions", len(train))
        logger.info("Validation set: %d transactions, class counts %s", len(validation), validation.class_counts())
        return train, validation

    @staticmethod
    def _ensure_matching_features(train: TransactionSet, validation: TransactionSet) -> None:
        if train.feature_names == validation.feature_names:
            return
        missing = [name for name in train.feature_names if name not in validation.feature_names]
        extra = [name for name in validation.feature_names if name not in train.feature_names]
        raise DataValidationError(
            "Training and validation feature columns differ: "
            f"missing in validation {missing}, unexpected in validation {extra}"
            + ("" if missing or extra else ", column order differs")
        )

    def _build_trainer(self, model: DeepAutoencoder, device: torch.device) -> AutoencoderTrainer:
        optimizer = create_optimizer(model.parameters(), self._config.optimizer)
        return AutoencoderTrainer(
            model=model,
            optimizer=optimizer,
            early_stopping=EarlyStopping.from_config(self._config.early_stopping),
            config=self._config.training,
            device=device,
            scheduler=create_scheduler(optimizer, self._config.scheduler),
        )

    def _build_monitor(self, validation: TransactionSet, device: torch.device) -> ReconstructionMonitor:
        scorer = ReconstructionErrorScorer(device, self._config.training.batch_size)
        return ReconstructionMonitor(
            scorer,
            validation,
            self._config.data.legitimate_label,
            self._config.evaluation.max_false_positive_rate,
        )

    def _train_loader(self, train: TransactionSet, device: torch.device) -> DataLoader:
        batch_size = self._config.training.batch_size
        return DataLoader(
            train.to_tensor_dataset(),
            batch_size=batch_size,
            shuffle=True,
            # BatchNorm cannot train on a trailing batch of a single sample
            drop_last=len(train) > batch_size,
            num_workers=self._config.training.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(self._config.seed),
        )

    @staticmethod
    def _log_summary(history: TrainingHistory) -> None:
        best = history.best_record
        if best is None:
            logger.warning("No epoch improved the validation loss")
            return
        logger.info(
            "Best epoch %d: val legit MSE %.6f, val fraud MSE %s, val fraud recall %s",
            best.epoch,
            best.validation_loss,
            "n/a" if best.validation_fraud_mse is None else f"{best.validation_fraud_mse:.6f}",
            "n/a" if best.validation_fraud_recall is None else f"{best.validation_fraud_recall:.1%}",
        )

    def _save_artifacts(self, model: DeepAutoencoder, feature_names: Tuple[str, ...], history: TrainingHistory) -> None:
        self._artifacts.save_model(model, feature_names)
        self._artifacts.save_history(history)
        self._artifacts.save_config(self._config)
        logger.info("Artifacts saved to %s", self._artifacts.directory)
