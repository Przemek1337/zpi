from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.exceptions import DataLoadingError, DataValidationError
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.validation_messages import describe_validation_error

logger = get_logger(__name__)

# split name in this project -> key suffix written by data/data_preprocessing.ipynb
_SPLIT_KEYS = {"train": "train", "validation": "val", "test": "test"}
_FEATURE_NAMES_KEY = "feature_names"


class NpzSplitsLoader:
    """Reads the train/validation splits archive (``splits.npz``) written by the preprocessing notebook."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def load(self, path: Path) -> Tuple[TransactionSet, TransactionSet]:
        train, validation = self.load_splits(path, ("train", "validation"))
        logger.debug("Loaded %s: train %d rows, validation %d rows", path, len(train), len(validation))
        return train, validation

    def load_splits(self, path: Path, splits: Tuple[str, ...]) -> Tuple[TransactionSet, ...]:
        path = Path(path)
        archive = self._read_archive(path)
        feature_names = self._feature_names(archive, path)
        keep = [index for index, name in enumerate(feature_names) if name not in self._config.drop_columns]
        kept_names = tuple(feature_names[index] for index in keep)
        return tuple(self._split(archive, split, keep, kept_names, path) for split in splits)

    @staticmethod
    def _read_archive(path: Path) -> Mapping[str, np.ndarray]:
        if not path.is_file():
            raise DataLoadingError(f"Splits archive not found: {path.absolute()}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise DataLoadingError(f"Cannot read splits archive {path}: {error}") from error

    def _feature_names(self, archive: Mapping[str, np.ndarray], path: Path) -> Tuple[str, ...]:
        if _FEATURE_NAMES_KEY not in archive:
            raise DataValidationError(f"{path}: missing '{_FEATURE_NAMES_KEY}' array")
        names = tuple(str(name) for name in archive[_FEATURE_NAMES_KEY].tolist())
        missing = [column for column in self._config.drop_columns if column not in names]
        if missing:
            raise DataValidationError(f"{path}: drop_columns not present in feature_names {missing}")
        return names

    @staticmethod
    def _split(
        archive: Mapping[str, np.ndarray],
        split: str,
        keep: list,
        feature_names: Tuple[str, ...],
        path: Path,
    ) -> TransactionSet:
        features_key, labels_key = f"X_{_SPLIT_KEYS[split]}", f"y_{_SPLIT_KEYS[split]}"
        missing = [key for key in (features_key, labels_key) if key not in archive]
        if missing:
            raise DataValidationError(f"{path}: missing arrays {missing}")
        features, labels = archive[features_key], archive[labels_key]
        if features.ndim != 2 or features.shape[1] != len(archive[_FEATURE_NAMES_KEY]):
            raise DataValidationError(
                f"{path}: {features_key} has shape {features.shape}, "
                f"expected (rows, {len(archive[_FEATURE_NAMES_KEY])})"
            )
        if len(features) == 0:
            raise DataValidationError(f"{path}: {split} split contains no rows")
        if not np.isfinite(features).all():
            raise DataValidationError(f"{path}: {features_key} contains missing or infinite values")
        if not np.issubdtype(labels.dtype, np.integer):
            raise DataValidationError(f"{path}: {labels_key} must contain integer class ids, got {labels.dtype}")
        try:
            return TransactionSet(
                features=np.ascontiguousarray(features[:, keep], dtype=np.float32),
                labels=labels.astype(np.int64),
                feature_names=feature_names,
            )
        except ValidationError as error:
            raise DataValidationError(
                f"{path}: inconsistent {split} split\n{describe_validation_error(error)}"
            ) from error
