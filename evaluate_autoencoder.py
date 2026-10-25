import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from fraud_detection.configuration.yaml_config_loader import YamlConfigLoader
from fraud_detection.evaluation.evaluation_plots import save_evaluation_plots
from fraud_detection.evaluation.evaluation_report import JSON_FILE, write_report
from fraud_detection.evaluation.test_evaluation import TestEvaluationReport, TestEvaluator
from fraud_detection.exceptions import ConfigurationError, FraudDetectionError
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.utils.colored_logging import configure_logging, get_logger
from train_autoencoder import LOG_LEVELS, ExitCode

DEFAULT_MODEL_DIRECTORY = Path("artifacts/tuning/final_model")
OUTPUT_SUBDIRECTORY = "test_evaluation"
BANNER_WIDTH = 78

logger = get_logger("evaluate_autoencoder")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained autoencoder once on the held-out test split (metrics, baseline, report)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
        help="directory with model.pt and experiment_config.yaml written by training",
    )
    parser.add_argument("--output", type=Path, default=None, help=f"report directory (default: <model-dir>/{OUTPUT_SUBDIRECTORY})")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default=None, help="overrides logging.level from config")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> ExitCode:
    arguments = parse_arguments(argv)
    configure_logging(arguments.log_level or "INFO")
    output = arguments.output or arguments.model_dir / OUTPUT_SUBDIRECTORY
    try:
        config_path = arguments.model_dir / ModelArtifactStore.CONFIG_FILE
        if not config_path.is_file():
            raise ConfigurationError(f"No {ModelArtifactStore.CONFIG_FILE} in {arguments.model_dir.absolute()}")
        config = YamlConfigLoader().load(config_path)
        configure_logging(arguments.log_level or config.logging.level, output / "evaluation.log")
        _warn_if_already_evaluated(output)
        report, scores, test = TestEvaluator(arguments.model_dir, config).run()
        plots = save_evaluation_plots(
            report,
            scores,
            test,
            config.data.legitimate_label,
            arguments.model_dir / ModelArtifactStore.HISTORY_FILE,
            output,
        )
        markdown = write_report(report, plots, output)
        _log_summary(report, markdown)
    except FraudDetectionError as error:
        logger.error("%s: %s", type(error).__name__, error, exc_info=logger.isEnabledFor(logging.DEBUG))
        return ExitCode.KNOWN_ERROR
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return ExitCode.INTERRUPTED
    except Exception:
        logger.critical("Unexpected error", exc_info=True)
        return ExitCode.UNEXPECTED_ERROR
    return ExitCode.SUCCESS


def _warn_if_already_evaluated(output: Path) -> None:
    if (output / JSON_FILE).is_file():
        logger.warning(
            "%s already holds a test evaluation. Regenerating the report is fine, but changing the model "
            "or threshold based on test results would make the test set no longer independent.",
            output,
        )


def _log_summary(report: TestEvaluationReport, markdown: Path) -> None:
    ae, base = report.autoencoder, report.baseline
    met = report.meets_recall and report.meets_fpr
    logger.info("=" * BANNER_WIDTH)
    logger.info("TEST EVALUATION - target: recall >= %.0f%% at FPR < %.0f%%",
                report.target_recall * 100, report.max_false_positive_rate * 100)
    for result in (ae, base):
        logger.info(
            "%-18s recall %5.1f%% | FPR %.2f%% | precision %5.1f%% | F1 %.3f | PR-AUC %.3f | TP %d FN %d FP %d TN %d",
            result.name, result.recall * 100, result.fpr * 100, result.precision * 100, result.f1, result.pr_auc,
            result.tp, result.fn, result.fp, result.tn,
        )
    logger.info("Missed frauds: %d (%d also missed by the baseline)",
                report.false_negatives.missed, report.false_negatives.missed_also_by_baseline)
    (logger.info if met else logger.warning)(
        "Specification criterion %s (recall %s, FPR %s)",
        "MET" if met else "NOT MET",
        "ok" if report.meets_recall else "below target",
        "ok" if report.meets_fpr else "above limit",
    )
    logger.info("Report: %s", markdown)
    logger.info("=" * BANNER_WIDTH)


if __name__ == "__main__":
    sys.exit(main())
