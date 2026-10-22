from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.utils.colored_logging import reset_logging

FEATURE_NAMES = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount"]


def make_transactions(legitimate: int, fraud: int, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    legitimate_rows = generator.normal(0.0, 1.0, size=(legitimate, len(FEATURE_NAMES)))
    fraud_rows = generator.normal(4.0, 3.0, size=(fraud, len(FEATURE_NAMES)))
    frame = pd.DataFrame(np.vstack([legitimate_rows, fraud_rows]), columns=FEATURE_NAMES)
    frame["Class"] = [0] * legitimate + [1] * fraud
    return frame


def make_transaction_set(legitimate: int, fraud: int, feature_count: int = 10) -> TransactionSet:
    generator = np.random.default_rng(1)
    features = np.vstack([
        generator.normal(0.0, 1.0, size=(legitimate, feature_count)),
        generator.normal(5.0, 3.0, size=(fraud, feature_count)),
    ]).astype(np.float32)
    labels = np.array([0] * legitimate + [1] * fraud, dtype=np.int64)
    return TransactionSet(features=features, labels=labels, feature_names=tuple(f"f{i}" for i in range(feature_count)))


@pytest.fixture(autouse=True)
def isolated_logging():
    yield
    reset_logging()


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[[str, pd.DataFrame], Path]:
    def write(name: str, frame: pd.DataFrame) -> Path:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    return write


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict], Path]:
    def write(raw: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    return write


@pytest.fixture
def experiment_dict(tmp_path: Path, write_csv) -> dict:
    train_path = write_csv("train.csv", make_transactions(legitimate=300, fraud=5, seed=1))
    validation_path = write_csv("validation.csv", make_transactions(legitimate=100, fraud=10, seed=2))
    return {
        "seed": 7,
        "device": "cpu",
        "data": {"train_path": str(train_path), "validation_path": str(validation_path), "drop_columns": ["Time"]},
        "model": {"hidden_dims": [16], "latent_dim": 4},
        "training": {"epochs": 3, "batch_size": 64},
        "output": {"directory": str(tmp_path / "artifacts")},
    }
