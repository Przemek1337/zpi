from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.exceptions import ConfigurationError
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.validation_messages import describe_validation_error

logger = get_logger(__name__)


class YamlConfigLoader:
    def load(self, path: Path) -> ExperimentConfig:
        raw = self._read_mapping(Path(path))
        try:
            config = ExperimentConfig.model_validate(raw)
        except ValidationError as error:
            raise ConfigurationError(
                f"Invalid configuration in {path}:\n{describe_validation_error(error)}"
            ) from error
        logger.debug("Configuration loaded from %s", path)
        return config

    @staticmethod
    def _read_mapping(path: Path) -> Dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as stream:
                content = yaml.safe_load(stream)
        except FileNotFoundError as error:
            raise ConfigurationError(f"Configuration file not found: {path.absolute()}") from error
        except OSError as error:
            raise ConfigurationError(f"Cannot read configuration file {path}: {error}") from error
        except yaml.YAMLError as error:
            raise ConfigurationError(f"Malformed YAML in {path}: {error}") from error
        if not isinstance(content, dict):
            raise ConfigurationError(
                f"Configuration root in {path} must be a mapping, got {type(content).__name__}"
            )
        return content
