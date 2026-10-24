import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.configuration.yaml_config_loader import YamlConfigLoader
from fraud_detection.exceptions import FraudDetectionError
from fraud_detection.tuning.tuning_config import TuningConfig
from fraud_detection.tuning.tuning_runner import TuningRunner
from fraud_detection.utils.colored_logging import configure_logging, get_logger
from train_autoencoder import LOG_LEVELS, ExitCode

DEFAULT_CONFIG_PATH = Path("config/autoencoder_tuning.yaml")
LOG_FILE_NAME = "tuning.log"

logger = get_logger("tune_autoencoder")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search autoencoder hyperparameters with Optuna, then train the best configuration"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to YAML tuning config")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default=None, help="overrides logging.level from config")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> ExitCode:
    arguments = parse_arguments(argv)
    configure_logging(arguments.log_level or "INFO")
    try:
        tuning = YamlConfigLoader(TuningConfig).load(arguments.config)
        base = YamlConfigLoader(ExperimentConfig).load(tuning.base_config)
        log_level = arguments.log_level or base.logging.level
        log_file = tuning.output.directory / LOG_FILE_NAME
        configure_logging(log_level, log_file)
        logger.info("Tuning config %s, base experiment config %s", arguments.config, tuning.base_config)
        TuningRunner(tuning, base, log_level, log_file).run()
    except FraudDetectionError as error:
        logger.error("%s: %s", type(error).__name__, error, exc_info=logger.isEnabledFor(logging.DEBUG))
        return ExitCode.KNOWN_ERROR
    except KeyboardInterrupt:
        logger.warning("Tuning interrupted by user - re-run the same command to resume the study")
        return ExitCode.INTERRUPTED
    except Exception:
        logger.critical("Unexpected error", exc_info=True)
        return ExitCode.UNEXPECTED_ERROR
    return ExitCode.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
