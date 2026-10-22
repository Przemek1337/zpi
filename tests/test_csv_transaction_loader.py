import numpy as np
import pytest

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.csv_transaction_loader import CsvTransactionLoader
from fraud_detection.exceptions import DataLoadingError, DataValidationError
from tests.conftest import FEATURE_NAMES, make_transactions


def loader(drop_columns=()) -> CsvTransactionLoader:
    return CsvTransactionLoader(DataConfig(train_path="unused", validation_path="unused", drop_columns=drop_columns))


def test_load_separates_features_and_labels(write_csv):
    path = write_csv("data.csv", make_transactions(legitimate=8, fraud=2))

    transactions = loader().load(path)

    assert transactions.features.shape == (10, len(FEATURE_NAMES))
    assert transactions.features.dtype == np.float32
    assert transactions.feature_names == tuple(FEATURE_NAMES)
    assert transactions.class_counts() == {0: 8, 1: 2}


def test_load_drops_configured_columns(write_csv):
    path = write_csv("data.csv", make_transactions(legitimate=4, fraud=0))

    transactions = loader(drop_columns=["Time"]).load(path)

    assert "Time" not in transactions.feature_names
    assert transactions.feature_count == len(FEATURE_NAMES) - 1


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(DataLoadingError, match="not found"):
        loader().load(tmp_path / "missing.csv")


def test_empty_file_is_reported(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadingError, match="empty"):
        loader().load(path)


def test_missing_label_column_is_reported(write_csv):
    path = write_csv("data.csv", make_transactions(legitimate=3, fraud=0).drop(columns=["Class"]))

    with pytest.raises(DataValidationError, match="Class"):
        loader().load(path)


def test_missing_drop_column_is_reported(write_csv):
    path = write_csv("data.csv", make_transactions(legitimate=3, fraud=0).drop(columns=["Time"]))

    with pytest.raises(DataValidationError, match="Time"):
        loader(drop_columns=["Time"]).load(path)


def test_header_only_file_is_reported(write_csv):
    path = write_csv("data.csv", make_transactions(legitimate=0, fraud=0))

    with pytest.raises(DataValidationError, match="no feature data"):
        loader().load(path)


def test_missing_values_name_the_column(write_csv):
    frame = make_transactions(legitimate=3, fraud=0)
    frame.loc[0, "V7"] = np.nan

    with pytest.raises(DataValidationError, match="V7"):
        loader().load(write_csv("data.csv", frame))


def test_infinite_values_name_the_column(write_csv):
    frame = make_transactions(legitimate=3, fraud=0)
    frame.loc[1, "Amount"] = np.inf

    with pytest.raises(DataValidationError, match="Amount"):
        loader().load(write_csv("data.csv", frame))


def test_non_numeric_features_are_reported(write_csv):
    frame = make_transactions(legitimate=3, fraud=0)
    frame["Merchant"] = "shop"

    with pytest.raises(DataValidationError, match="Merchant"):
        loader().load(write_csv("data.csv", frame))


@pytest.mark.parametrize("bad_label", [0.5, "fraud", np.nan])
def test_invalid_labels_are_reported(write_csv, bad_label):
    frame = make_transactions(legitimate=3, fraud=0).astype({"Class": object})
    frame.loc[0, "Class"] = bad_label

    with pytest.raises(DataValidationError, match="label column"):
        loader().load(write_csv("data.csv", frame))
