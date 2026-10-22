from pathlib import Path

import pytest

from fraud_detection.configuration.yaml_config_loader import YamlConfigLoader
from fraud_detection.exceptions import ConfigurationError

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "autoencoder_training.yaml"


def test_project_config_loads():
    config = YamlConfigLoader().load(PROJECT_CONFIG)

    assert isinstance(config.optimizer.learning_rate, float)
    assert config.model.hidden_dims == (24, 16)
    assert config.data.label_column == "Class"


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        YamlConfigLoader().load(tmp_path / "missing.yaml")


def test_malformed_yaml_is_reported(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("model: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Malformed YAML"):
        YamlConfigLoader().load(path)


def test_non_mapping_root_is_reported(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be a mapping"):
        YamlConfigLoader().load(path)


def test_validation_errors_point_to_field(write_config):
    path = write_config({"data": {"train_path": "a", "validation_path": "b"}, "model": {"dropout": 2}})

    with pytest.raises(ConfigurationError, match=r"model\.dropout"):
        YamlConfigLoader().load(path)
