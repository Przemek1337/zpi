from __future__ import annotations

import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import torch
import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from fraud_detection.configuration.experiment_config import ExperimentConfig, ModelConfig
from fraud_detection.exceptions import ArtifactError, ConfigurationError
from fraud_detection.model.deep_autoencoder import DeepAutoencoder
from fraud_detection.training.training_history import TrainingHistory
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.validation_messages import describe_validation_error

logger = get_logger(__name__)


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_dim: PositiveInt
    architecture: ModelConfig
    feature_names: Tuple[str, ...]


class ModelArtifactStore:
    MODEL_FILE = "model.pt"
    HISTORY_FILE = "training_history.json"
    CONFIG_FILE = "experiment_config.yaml"

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def save_model(self, model: DeepAutoencoder, feature_names: Sequence[str]) -> Path:
        metadata = CheckpointMetadata(
            input_dim=model.input_dim,
            architecture=model.config,
            feature_names=tuple(feature_names),
        )
        path = self._prepare(self.MODEL_FILE)
        with _artifact_operation(f"save model checkpoint to {path}"):
            torch.save({"metadata": metadata.model_dump(mode="json"), "state_dict": model.state_dict()}, path)
        logger.debug("Model checkpoint saved to %s", path)
        return path

    def load_model(self, device: torch.device) -> Tuple[DeepAutoencoder, Tuple[str, ...]]:
        path = self._directory / self.MODEL_FILE
        if not path.is_file():
            raise ArtifactError(f"Model checkpoint not found: {path.absolute()}")
        with _artifact_operation(f"read model checkpoint {path}"):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        metadata = self._parse_metadata(checkpoint, path)
        try:
            model = DeepAutoencoder(metadata.input_dim, metadata.architecture)
        except ConfigurationError as error:
            raise ArtifactError(f"Checkpoint {path} describes an invalid model: {error}") from error
        with _artifact_operation(f"restore weights from {path}"):
            model.load_state_dict(checkpoint["state_dict"])
        logger.debug("Model checkpoint loaded from %s", path)
        return model.to(device).eval(), metadata.feature_names

    def save_history(self, history: TrainingHistory) -> Path:
        path = self._prepare(self.HISTORY_FILE)
        with _artifact_operation(f"save training history to {path}"):
            path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
        return path

    def save_config(self, config: ExperimentConfig) -> Path:
        path = self._prepare(self.CONFIG_FILE)
        with _artifact_operation(f"save experiment config to {path}"):
            path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _parse_metadata(checkpoint: Any, path: Path) -> CheckpointMetadata:
        if not isinstance(checkpoint, dict) or not {"metadata", "state_dict"} <= checkpoint.keys():
            raise ArtifactError(f"Checkpoint {path} has unexpected structure")
        try:
            return CheckpointMetadata.model_validate(checkpoint["metadata"])
        except ValidationError as error:
            raise ArtifactError(
                f"Checkpoint {path} has invalid metadata:\n{describe_validation_error(error)}"
            ) from error

    def _prepare(self, file_name: str) -> Path:
        with _artifact_operation(f"create artifact directory {self._directory}"):
            self._directory.mkdir(parents=True, exist_ok=True)
        return self._directory / file_name


@contextmanager
def _artifact_operation(description: str) -> Iterator[None]:
    try:
        yield
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise ArtifactError(f"Failed to {description}: {error}") from error
