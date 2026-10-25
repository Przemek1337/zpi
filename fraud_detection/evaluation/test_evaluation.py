from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.data.dataset_source import load_test
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.detection_metrics import threshold_for_fpr
from fraud_detection.evaluation.isolation_forest_baseline import IsolationForestBaseline
from fraud_detection.evaluation.reconstruction_error_scorer import ReconstructionErrorScorer
from fraud_detection.exceptions import DataValidationError
from fraud_detection.model.deep_autoencoder import DeepAutoencoder
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.training.training_runner import TrainingRunner
from fraud_detection.utils.colored_logging import get_logger
from fraud_detection.utils.device_resolver import resolve_device
from fraud_detection.utils.reproducibility import seed_everything

logger = get_logger(__name__)

TARGET_RECALL = 0.85
SCORING_BATCH_SIZE = 4096
HARDEST_CASES = 10
TOP_FEATURES = 6


class DetectionResult(BaseModel):
    """Detector quality on the test set at the threshold chosen on validation data."""

    model_config = ConfigDict(frozen=True)

    name: str
    threshold: float
    validation_recall: float
    validation_fpr: float
    tn: int
    fp: int
    fn: int
    tp: int
    recall: float
    fpr: float
    precision: float
    f1: float
    roc_auc: float
    pr_auc: float


class HardCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_row: int
    mse: float
    share_of_threshold: float
    legitimate_percentile: float
    detected_by_baseline: bool


class FeatureError(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    detected_fraud: float
    missed_fraud: float
    legitimate: float


class FalseNegativeAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    missed: int
    detected: int
    missed_also_by_baseline: int
    median_mse_missed: Optional[float]
    median_mse_detected: Optional[float]
    median_mse_legitimate: float
    hardest_cases: List[HardCase]
    feature_errors: List[FeatureError]


class Timing(BaseModel):
    model_config = ConfigDict(frozen=True)

    device: str
    autoencoder_training_seconds: Optional[float]
    autoencoder_test_prediction_seconds: float
    baseline_fit_seconds: float
    baseline_test_prediction_seconds: float
    peak_gpu_memory_mb: Optional[float]


class TestEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluated_at: str
    model_directory: str
    max_false_positive_rate: float
    target_recall: float
    test_legitimate: int
    test_fraud: int
    autoencoder: DetectionResult
    baseline: DetectionResult
    false_negatives: FalseNegativeAnalysis
    timing: Timing
    model_parameters: int
    experiment: Dict

    @property
    def meets_recall(self) -> bool:
        return self.autoencoder.recall >= self.target_recall

    @property
    def meets_fpr(self) -> bool:
        return self.autoencoder.fpr < self.max_false_positive_rate


class ScoredSplits(BaseModel):
    """Scores of one detector on validation and test, kept for plots."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    validation: np.ndarray
    test: np.ndarray


class TestEvaluator:
    """One-off evaluation of a trained autoencoder on the held-out test split.

    The decision threshold is derived from validation data only and then applied unchanged to the
    test split, so no test information leaks into any choice made about the model.
    """

    def __init__(self, model_directory: Path, config: ExperimentConfig) -> None:
        self._model_directory = Path(model_directory)
        self._config = config

    def run(self) -> Tuple[TestEvaluationReport, Dict[str, ScoredSplits], TransactionSet]:
        seed_everything(self._config.seed)
        device = resolve_device(self._config.device)
        model, feature_names = ModelArtifactStore(self._model_directory).load_model(device)
        logger.info("Loaded model from %s (%d parameters) on %s", self._model_directory, model.parameter_count, device)
        train, validation, test = self._load_data(feature_names)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        autoencoder_scores, prediction_seconds = self._score_autoencoder(model, validation, test, device)
        baseline = IsolationForestBaseline(seed=self._config.seed).fit(train)
        baseline_scores = ScoredSplits(validation=baseline.score(validation), test=baseline.score(test))
        logger.info("Isolation Forest baseline fitted in %.1fs", baseline.fit_seconds)

        max_fpr = self._config.evaluation.max_false_positive_rate
        autoencoder = evaluate_detector("Autoencoder (MSE)", autoencoder_scores, validation, test, self._legit, max_fpr)
        reference = evaluate_detector("Isolation Forest", baseline_scores, validation, test, self._legit, max_fpr)
        report = TestEvaluationReport(
            evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_directory=str(self._model_directory),
            max_false_positive_rate=max_fpr,
            target_recall=TARGET_RECALL,
            test_legitimate=int(np.sum(test.labels == self._legit)),
            test_fraud=int(np.sum(test.labels != self._legit)),
            autoencoder=autoencoder,
            baseline=reference,
            false_negatives=self._analyse_false_negatives(
                model, test, autoencoder, reference, autoencoder_scores, baseline_scores, device
            ),
            timing=Timing(
                device=str(device) + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
                autoencoder_training_seconds=self._training_seconds(),
                autoencoder_test_prediction_seconds=prediction_seconds,
                baseline_fit_seconds=baseline.fit_seconds,
                baseline_test_prediction_seconds=baseline.predict_seconds,
                peak_gpu_memory_mb=(
                    torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
                ),
            ),
            model_parameters=model.parameter_count,
            experiment=self._config.model_dump(mode="json"),
        )
        return report, {"autoencoder": autoencoder_scores, "baseline": baseline_scores}, test

    @property
    def _legit(self) -> int:
        return self._config.data.legitimate_label

    def _load_data(self, feature_names: Tuple[str, ...]) -> Tuple[TransactionSet, TransactionSet, TransactionSet]:
        train, validation = TrainingRunner(self._config).load_datasets()
        test = load_test(self._config.data)
        for name, split in (("train", train), ("validation", validation), ("test", test)):
            if split.feature_names != tuple(feature_names):
                raise DataValidationError(
                    f"{name} features {list(split.feature_names)} differ from the model's {list(feature_names)}"
                )
        if np.all(test.labels == self._legit) or np.all(validation.labels == self._legit):
            raise DataValidationError("Validation and test splits must both contain fraud transactions")
        logger.info(
            "Test set: %d transactions, class counts %s (threshold is chosen on validation only)",
            len(test),
            test.class_counts(),
        )
        return train, validation, test

    @staticmethod
    def _score_autoencoder(
        model: DeepAutoencoder, validation: TransactionSet, test: TransactionSet, device: torch.device
    ) -> Tuple[ScoredSplits, float]:
        scorer = ReconstructionErrorScorer(device, SCORING_BATCH_SIZE)
        scorer.score(model, test.features[:SCORING_BATCH_SIZE])  # warm-up, excluded from the timing
        validation_scores = scorer.score(model, validation.features)
        _synchronize(device)
        started = time.perf_counter()
        test_scores = scorer.score(model, test.features)
        _synchronize(device)
        seconds = time.perf_counter() - started
        logger.info("Autoencoder scored %d test transactions in %.3fs", len(test), seconds)
        return ScoredSplits(validation=validation_scores, test=test_scores), seconds

    def _analyse_false_negatives(
        self,
        model: DeepAutoencoder,
        test: TransactionSet,
        autoencoder: DetectionResult,
        baseline: DetectionResult,
        autoencoder_scores: ScoredSplits,
        baseline_scores: ScoredSplits,
        device: torch.device,
    ) -> FalseNegativeAnalysis:
        scores = autoencoder_scores.test
        fraud = test.labels != self._legit
        detected = fraud & (scores > autoencoder.threshold)
        missed = fraud & ~detected
        baseline_detected = baseline_scores.test > baseline.threshold
        legitimate_scores = scores[~fraud]
        fraud_rows = np.flatnonzero(fraud)
        hardest = fraud_rows[np.argsort(scores[fraud_rows])][:HARDEST_CASES]
        return FalseNegativeAnalysis(
            missed=int(missed.sum()),
            detected=int(detected.sum()),
            missed_also_by_baseline=int((missed & ~baseline_detected).sum()),
            median_mse_missed=_median(scores[missed]),
            median_mse_detected=_median(scores[detected]),
            median_mse_legitimate=float(np.median(legitimate_scores)),
            hardest_cases=[
                HardCase(
                    test_row=int(row),
                    mse=float(scores[row]),
                    share_of_threshold=float(scores[row] / autoencoder.threshold),
                    legitimate_percentile=float(np.mean(legitimate_scores < scores[row]) * 100),
                    detected_by_baseline=bool(baseline_detected[row]),
                )
                for row in hardest
            ],
            feature_errors=self._feature_errors(model, test, detected, missed, device),
        )

    def _feature_errors(
        self, model: DeepAutoencoder, test: TransactionSet, detected: np.ndarray, missed: np.ndarray, device
    ) -> List[FeatureError]:
        def mean_squared_error_per_feature(mask: np.ndarray) -> Optional[np.ndarray]:
            if not mask.any():
                return None
            errors = []
            with torch.no_grad():
                for start in range(0, int(mask.sum()), SCORING_BATCH_SIZE):
                    batch = torch.from_numpy(test.features[mask][start : start + SCORING_BATCH_SIZE]).to(device)
                    errors.append(((model(batch) - batch) ** 2).cpu().numpy())
            return np.concatenate(errors).mean(axis=0)

        detected_errors = mean_squared_error_per_feature(detected)
        if detected_errors is None:
            return []
        missed_errors = mean_squared_error_per_feature(missed)
        legitimate_errors = mean_squared_error_per_feature(test.labels == self._legit)
        top = np.argsort(detected_errors)[::-1][:TOP_FEATURES]
        return [
            FeatureError(
                feature=test.feature_names[index],
                detected_fraud=float(detected_errors[index]),
                missed_fraud=float(missed_errors[index]) if missed_errors is not None else 0.0,
                legitimate=float(legitimate_errors[index]),
            )
            for index in top
        ]

    def _training_seconds(self) -> Optional[float]:
        history = self._model_directory / ModelArtifactStore.HISTORY_FILE
        if not history.is_file():
            return None
        return float(json.loads(history.read_text(encoding="utf-8")).get("total_seconds", 0.0)) or None


def evaluate_detector(
    name: str,
    scores: ScoredSplits,
    validation: TransactionSet,
    test: TransactionSet,
    legitimate_label: int,
    max_fpr: float,
) -> DetectionResult:
    validation_fraud = validation.labels != legitimate_label
    threshold = threshold_for_fpr(scores.validation[~validation_fraud], max_fpr)
    fraud = test.labels != legitimate_label
    flagged = scores.test > threshold
    tp, fn = int(np.sum(flagged & fraud)), int(np.sum(~flagged & fraud))
    fp, tn = int(np.sum(flagged & ~fraud)), int(np.sum(~flagged & ~fraud))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return DetectionResult(
        name=name,
        threshold=threshold,
        validation_recall=float(np.mean(scores.validation[validation_fraud] > threshold)),
        validation_fpr=float(np.mean(scores.validation[~validation_fraud] > threshold)),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        recall=recall,
        fpr=fp / (fp + tn),
        precision=precision,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        roc_auc=float(roc_auc_score(fraud, scores.test)),
        pr_auc=float(average_precision_score(fraud, scores.test)),
    )


def _median(values: np.ndarray) -> Optional[float]:
    return float(np.median(values)) if len(values) else None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
