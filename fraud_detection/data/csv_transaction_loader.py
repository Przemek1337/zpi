from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.exceptions import DataLoadingError, DataValidationError
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.validation_messages import describe_validation_error

logger = get_logger(__name__)


class CsvTransactionLoader:
    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def load(self, path: Path) -> TransactionSet:
        path = Path(path)
        frame = self._read_csv(path)
        self._ensure_columns_present(frame, path)
        features = frame.drop(columns=[self._config.label_column, *self._config.drop_columns])
        labels = frame[self._config.label_column]
        self._ensure_features_valid(features, path)
        self._ensure_labels_valid(labels, path)
        transactions = self._build_transaction_set(features, labels, path)
        logger.debug(
            "Loaded %s: %d rows, %d features, class counts %s",
            path,
            len(transactions),
            transactions.feature_count,
            transactions.class_counts(),
        )
        return transactions

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise DataLoadingError(f"CSV file not found: {path.absolute()}")
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError as error:
            raise DataLoadingError(f"CSV file is empty: {path}") from error
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            raise DataLoadingError(f"Cannot parse CSV file {path}: {error}") from error
        except OSError as error:
            raise DataLoadingError(f"Cannot read CSV file {path}: {error}") from error

    def _ensure_columns_present(self, frame: pd.DataFrame, path: Path) -> None:
        required = [self._config.label_column, *self._config.drop_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DataValidationError(f"{path}: missing required columns {missing}")

    @staticmethod
    def _ensure_features_valid(features: pd.DataFrame, path: Path) -> None:
        if features.empty:
            raise DataValidationError(f"{path}: contains no feature data")
        non_numeric = features.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise DataValidationError(f"{path}: non-numeric feature columns {non_numeric}")
        with_missing = features.columns[features.isna().any()].tolist()
        if with_missing:
            raise DataValidationError(f"{path}: missing values in columns {with_missing}")
        with_infinite = features.columns[np.isinf(features.to_numpy()).any(axis=0)].tolist()
        if with_infinite:
            raise DataValidationError(f"{path}: infinite values in columns {with_infinite}")

    def _ensure_labels_valid(self, labels: pd.Series, path: Path) -> None:
        column = self._config.label_column
        if labels.isna().any():
            raise DataValidationError(f"{path}: label column '{column}' has {int(labels.isna().sum())} missing values")
        if not is_numeric_dtype(labels) or not np.all(np.mod(labels.to_numpy(), 1) == 0):
            raise DataValidationError(f"{path}: label column '{column}' must contain integer class ids")

    @staticmethod
    def _build_transaction_set(features: pd.DataFrame, labels: pd.Series, path: Path) -> TransactionSet:
        try:
            return TransactionSet(
                features=features.to_numpy(dtype=np.float32),
                labels=labels.to_numpy(dtype=np.int64),
                feature_names=tuple(str(column) for column in features.columns),
            )
        except ValidationError as error:
            raise DataValidationError(f"{path}: inconsistent data\n{describe_validation_error(error)}") from error
