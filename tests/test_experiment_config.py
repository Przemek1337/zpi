from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import ActivationName, ExperimentConfig, OptimizerName
from fraud_detection.model.activation_registry import SUPPORTED_ACTIVATIONS
from fraud_detection.training.optimizer_factory import SUPPORTED_OPTIMIZERS

MINIMAL = {"data": {"train_path": "a.csv", "validation_path": "b.csv"}}


def with_section(name: str, values: dict) -> dict:
    return {**MINIMAL, name: values}


def test_missing_sections_fall_back_to_defaults():
    config = ExperimentConfig.model_validate(MINIMAL)

    assert config.training.epochs == 100
    assert config.device == "auto"
    assert config.logging.level == "INFO"
    assert isinstance(config.data.train_path, Path)


def test_numeric_strings_are_coerced():
    config = ExperimentConfig.model_validate(with_section("optimizer", {"learning_rate": "1e-3"}))

    assert config.optimizer.learning_rate == pytest.approx(1e-3)


def test_config_is_immutable():
    config = ExperimentConfig.model_validate(MINIMAL)

    with pytest.raises(ValidationError):
        config.seed = 1


@pytest.mark.parametrize(
    "raw",
    [
        {"seed": 1},
        {**MINIMAL, "modle": {}},
        with_section("model", {"latent": 4}),
        with_section("model", {"dropout": 1.5}),
        with_section("model", {"activation": "swish"}),
        with_section("model", {"hidden_dims": [16, 8], "latent_dim": 8}),
        with_section("optimizer", {"learning_rate": 0}),
        with_section("optimizer", {"name": "lbfgs"}),
        with_section("training", {"batch_size": -1}),
        with_section("logging", {"level": "VERBOSE"}),
        {**MINIMAL, "device": "tpu"},
        {"data": {"train_path": "a", "validation_path": "b", "drop_columns": ["Class"]}},
    ],
)
def test_invalid_values_are_rejected(raw):
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_literal_names_match_registries():
    assert set(get_args(ActivationName)) == SUPPORTED_ACTIVATIONS
    assert set(get_args(OptimizerName)) == SUPPORTED_OPTIMIZERS
