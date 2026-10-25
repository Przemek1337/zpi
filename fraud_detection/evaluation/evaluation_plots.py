from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless servers have no display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.test_evaluation import DetectionResult, ScoredSplits, TestEvaluationReport

DPI = 130
LEGITIMATE_COLOR = "tab:blue"
FRAUD_COLOR = "tab:red"


def save_evaluation_plots(
    report: TestEvaluationReport,
    scores: Dict[str, ScoredSplits],
    test: TransactionSet,
    legitimate_label: int,
    history_path: Path,
    directory: Path,
) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fraud = test.labels != legitimate_label
    paths = [
        _mse_distribution(scores["autoencoder"].test, fraud, report.autoencoder.threshold, directory),
        _confusion_matrices([report.autoencoder, report.baseline], directory),
        _precision_recall(report, scores, fraud, directory),
    ]
    if history_path.is_file():
        paths.append(_learning_curves(history_path, directory))
    return paths


def _mse_distribution(scores: np.ndarray, fraud: np.ndarray, threshold: float, directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(max(scores.min(), 1e-6)), np.log10(scores.max()), 80)
    ax.hist(scores[~fraud], bins=bins, density=True, alpha=0.6, color=LEGITIMATE_COLOR, label="prawidłowe")
    ax.hist(scores[fraud], bins=bins, density=True, alpha=0.6, color=FRAUD_COLOR, label="fraud")
    ax.axvline(threshold, color="black", linestyle="--", label=f"próg {threshold:.3f} (z walidacji)")
    ax.set_xscale("log")
    ax.set_xlabel("MSE rekonstrukcji (skala log)")
    ax.set_ylabel("gęstość")
    ax.set_title("Rozkład błędu rekonstrukcji - zbiór testowy")
    ax.legend()
    return _save(fig, directory / "mse_distribution.png")


def _confusion_matrices(results: List[DetectionResult], directory: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    for ax, result in zip(np.atleast_1d(axes), results):
        matrix = np.array([[result.tn, result.fp], [result.fn, result.tp]])
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                ax.text(
                    column, row, f"{matrix[row, column]}\n({shares[row, column]:.1%})",
                    ha="center", va="center", color="white" if shares[row, column] > 0.5 else "black",
                )
        ax.set_xticks([0, 1], ["prawidłowa", "fraud"])
        ax.set_yticks([0, 1], ["prawidłowa", "fraud"])
        ax.set_xlabel("predykcja")
        ax.set_ylabel("rzeczywistość")
        ax.set_title(f"{result.name}\nrecall {result.recall:.1%} | FPR {result.fpr:.2%}")
    return _save(fig, directory / "confusion_matrices.png")


def _precision_recall(
    report: TestEvaluationReport, scores: Dict[str, ScoredSplits], fraud: np.ndarray, directory: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, result, color in (
        ("autoencoder", report.autoencoder, "tab:purple"),
        ("baseline", report.baseline, "tab:gray"),
    ):
        precision, recall, _ = precision_recall_curve(fraud, scores[key].test)
        ax.plot(recall, precision, color=color, label=f"{result.name} (PR-AUC {result.pr_auc:.3f})")
        ax.scatter([result.recall], [result.precision], color=color, zorder=3, edgecolor="black")
    ax.axvline(report.target_recall, color=FRAUD_COLOR, linestyle=":", label=f"cel recall {report.target_recall:.0%}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall na teście (kropki: progi z walidacji)")
    ax.legend(loc="upper right")
    return _save(fig, directory / "precision_recall.png")


def _learning_curves(history_path: Path, directory: Path) -> Path:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history["records"]
    epochs = [record["epoch"] for record in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [record["train_loss"] for record in records], label="train MSE")
    ax.plot(epochs, [record["validation_loss"] for record in records], label="walidacja MSE (prawidłowe)")
    ax.set_xlabel("epoka")
    ax.set_ylabel("MSE")
    recalls = [record.get("validation_fraud_recall") for record in records]
    if all(value is not None for value in recalls):
        twin = ax.twinx()
        twin.plot(epochs, recalls, color=FRAUD_COLOR, alpha=0.7, label="walidacja recall")
        twin.set_ylabel("recall (walidacja)")
        twin.legend(loc="center right")
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"], color="black", linestyle=":", label=f"zapisana epoka {history['best_epoch']}")
    ax.set_title("Krzywe uczenia modelu")
    ax.legend(loc="upper right")
    return _save(fig, directory / "learning_curves.png")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
