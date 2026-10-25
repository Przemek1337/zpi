import argparse
import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

from fraud_detection.configuration.yaml_config_loader import YamlConfigLoader
from fraud_detection.exceptions import FraudDetectionError
from fraud_detection.training.training_runner import TrainingRunner
from fraud_detection.utils.colored_logging import configure_logging, get_logger

DEFAULT_CONFIG_PATH = Path("config/autoencoder_training.yaml")
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

logger = get_logger("train_autoencoder")


class ExitCode(IntEnum):
    SUCCESS = 0
    KNOWN_ERROR = 1
    UNEXPECTED_ERROR = 2
    INTERRUPTED = 130


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep autoencoder for credit card fraud detection")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to YAML experiment config")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default=None, help="overrides logging.level from config")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> ExitCode:
    arguments = parse_arguments(argv)
    configure_logging(arguments.log_level or "INFO")
    try:
        config = YamlConfigLoader().load(arguments.config)
        configure_logging(arguments.log_level or config.logging.level, config.logging.file)
        TrainingRunner(config).run()
    except FraudDetectionError as error:
        logger.error("%s: %s", type(error).__name__, error, exc_info=logger.isEnabledFor(logging.DEBUG))
        return ExitCode.KNOWN_ERROR
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return ExitCode.INTERRUPTED
    except Exception:
        logger.critical("Unexpected error", exc_info=True)
        return ExitCode.UNEXPECTED_ERROR
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
