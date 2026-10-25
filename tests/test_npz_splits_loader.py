import numpy as np
import pytest

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.npz_splits_loader import NpzSplitsLoader
from fraud_detection.exceptions import DataLoadingError, DataValidationError
from tests.conftest import FEATURE_NAMES, make_transactions, save_splits


def loader(drop_columns=()) -> NpzSplitsLoader:
    return NpzSplitsLoader(DataConfig(splits_path="unused.npz", drop_columns=drop_columns))


@pytest.fixture
def splits_path(tmp_path):
    return save_splits(
        tmp_path / "splits.npz",
        make_transactions(legitimate=20, fraud=0, seed=1),
        make_transactions(legitimate=8, fraud=3, seed=2),
    )


def test_load_returns_train_and_validation(splits_path):
    train, validation = loader().load(splits_path)

    assert train.features.shape == (20, len(FEATURE_NAMES))
    assert train.features.dtype == np.float32
    assert train.feature_names == tuple(FEATURE_NAMES)
    assert validation.class_counts() == {0: 8, 1: 3}


def test_load_drops_configured_columns(splits_path):
    train, validation = loader(drop_columns=["Time"]).load(splits_path)

    assert "Time" not in train.feature_names
    assert train.feature_count == validation.feature_count == len(FEATURE_NAMES) - 1


def test_unknown_drop_column_is_reported(splits_path):
    with pytest.raises(DataValidationError, match="Merchant"):
        loader(drop_columns=["Merchant"]).load(splits_path)


def test_missing_archive_is_reported(tmp_path):
    with pytest.raises(DataLoadingError, match="not found"):
        loader().load(tmp_path / "missing.npz")


def test_corrupted_archive_is_reported(tmp_path):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(DataLoadingError, match="Cannot read"):
        loader().load(path)


def test_missing_arrays_are_reported(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez(path, feature_names=np.array(["a"]), X_train=np.zeros((2, 1), dtype=np.float32))

    with pytest.raises(DataValidationError, match="y_train"):
        loader().load(path)


def test_non_finite_features_are_reported(tmp_path):
    train = make_transactions(legitimate=5, fraud=0)
    train.loc[0, "V3"] = np.inf

    with pytest.raises(DataValidationError, match="X_train contains missing or infinite"):
        loader().load(save_splits(tmp_path / "splits.npz", train, make_transactions(3, 1)))


def test_float_labels_are_reported(tmp_path):
    path = tmp_path / "splits.npz"
    np.savez(
        path,
        feature_names=np.array(["a"]),
        X_train=np.zeros((2, 1), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.float32),
        X_val=np.zeros((2, 1), dtype=np.float32),
        y_val=np.zeros(2, dtype=np.int64),
    )

    with pytest.raises(DataValidationError, match="integer class ids"):
        loader().load(path)


def test_feature_count_mismatch_is_reported(tmp_path):
    path = tmp_path / "splits.npz"
    np.savez(
        path,
        feature_names=np.array(["a", "b"]),
        X_train=np.zeros((2, 3), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.int64),
        X_val=np.zeros((2, 2), dtype=np.float32),
        y_val=np.zeros(2, dtype=np.int64),
    )

    with pytest.raises(DataValidationError, match="X_train has shape"):
        loader().load(path)
