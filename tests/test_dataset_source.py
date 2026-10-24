import json
from importlib.util import find_spec

import pytest

from fraud_detection.configuration.experiment_config import DataConfig
from fraud_detection.data.dataset_source import ensure_prepared, load_train_validation
from fraud_detection.data.notebook_preparation import prepare_splits
from fraud_detection.exceptions import DataLoadingError
from tests.conftest import make_transactions, save_splits

notebook_runtime = pytest.mark.skipif(
    find_spec("nbclient") is None or find_spec("ipykernel") is None, reason="nbclient/ipykernel not installed"
)


def write_notebook(path, *cells):
    path.write_text(
        json.dumps({
            "cells": [
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}
                for source in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )
    return path


def test_csv_source_loads_both_files(write_csv):
    config = DataConfig(
        train_path=write_csv("train.csv", make_transactions(10, 0)),
        validation_path=write_csv("validation.csv", make_transactions(5, 2)),
    )

    train, validation = load_train_validation(config)

    assert len(train) == 10
    assert validation.class_counts() == {0: 5, 1: 2}


def test_existing_archive_is_not_regenerated(tmp_path):
    splits = save_splits(tmp_path / "splits.npz", make_transactions(10, 0), make_transactions(5, 2))
    notebook = write_notebook(tmp_path / "prep.ipynb", "raise RuntimeError('must not run')")

    assert ensure_prepared(DataConfig(splits_path=splits, preparation_notebook=notebook)) is False


def test_missing_archive_without_notebook_is_reported(tmp_path):
    with pytest.raises(DataLoadingError, match="Splits archive not found"):
        load_train_validation(DataConfig(splits_path=tmp_path / "missing.npz"))


def test_missing_notebook_is_reported(tmp_path):
    with pytest.raises(DataLoadingError, match="Preparation notebook not found"):
        prepare_splits(tmp_path / "splits.npz", tmp_path / "missing.ipynb")


@notebook_runtime
def test_notebook_creates_missing_archive(tmp_path):
    source = save_splits(tmp_path / "source.npz", make_transactions(10, 0), make_transactions(5, 2))
    (tmp_path / "processed").mkdir()
    # relative paths: the notebook must run inside its own directory, like data/data_preprocessing.ipynb
    notebook = write_notebook(
        tmp_path / "prep.ipynb",
        "import shutil\nshutil.copy('source.npz', 'processed/splits.npz')",
    )
    config = DataConfig(splits_path=tmp_path / "processed" / "splits.npz", preparation_notebook=notebook)

    train, validation = load_train_validation(config)

    assert source.is_file()
    assert len(train) == 10
    assert validation.class_counts() == {0: 5, 1: 2}


@notebook_runtime
def test_failing_notebook_is_reported(tmp_path):
    notebook = write_notebook(tmp_path / "prep.ipynb", "raise ValueError('broken preprocessing')")

    with pytest.raises(DataLoadingError, match="broken preprocessing"):
        prepare_splits(tmp_path / "splits.npz", notebook)


@notebook_runtime
def test_notebook_that_writes_nothing_is_reported(tmp_path):
    notebook = write_notebook(tmp_path / "prep.ipynb", "x = 1")

    with pytest.raises(DataLoadingError, match="did not create"):
        prepare_splits(tmp_path / "splits.npz", notebook)
