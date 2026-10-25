from __future__ import annotations

from typing import Tuple

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.csv_transaction_loader import CsvTransactionLoader
from fraud_detection.data.notebook_preparation import prepare_splits
from fraud_detection.data.npz_splits_loader import NpzSplitsLoader
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.exceptions import DataLoadingError


def ensure_prepared(config: DataConfig) -> bool:
    """Creates the splits archive from the preparation notebook when it is missing; returns True if it did."""
    if not config.uses_splits_archive or config.splits_path.is_file() or config.preparation_notebook is None:
        return False
    prepare_splits(config.splits_path, config.preparation_notebook)
    return True


def load_train_validation(config: DataConfig) -> Tuple[TransactionSet, TransactionSet]:
    if config.uses_splits_archive:
        ensure_prepared(config)
        return NpzSplitsLoader(config).load(config.splits_path)
    loader = CsvTransactionLoader(config)
    return loader.load(config.train_path), loader.load(config.validation_path)


def load_test(config: DataConfig) -> TransactionSet:
    """Held-out test split; only the final one-off evaluation may call this."""
    if config.uses_splits_archive:
        ensure_prepared(config)
        (test,) = NpzSplitsLoader(config).load_splits(config.splits_path, ("test",))
        return test
    if config.test_path is None:
        raise DataLoadingError("data.test_path is required to evaluate on the test set with CSV sources")
    return CsvTransactionLoader(config).load(config.test_path)
