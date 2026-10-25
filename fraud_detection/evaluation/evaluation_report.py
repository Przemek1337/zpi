from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fraud_detection.evaluation.test_evaluation import DetectionResult, TestEvaluationReport

JSON_FILE = "report.json"
MARKDOWN_FILE = "report.md"


def write_report(report: TestEvaluationReport, plots: List[Path], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        **report.model_dump(mode="json"),
        "criteria": {"recall_met": report.meets_recall, "fpr_met": report.meets_fpr},
    }
    (directory / JSON_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = directory / MARKDOWN_FILE
    markdown.write_text(render_markdown(report, plots), encoding="utf-8")
    return markdown


def render_markdown(report: TestEvaluationReport, plots: List[Path]) -> str:
    ae, base, fn, timing = report.autoencoder, report.baseline, report.false_negatives, report.timing
    model = report.experiment["model"]
    verdict = "SPEŁNIONE" if report.meets_recall and report.meets_fpr else "NIESPEŁNIONE"
    lines = [
        "# Ewaluacja autoenkodera na zbiorze testowym",
        "",
        f"Model: `{report.model_directory}` | ocena: {report.evaluated_at} | urządzenie: {timing.device}",
        "",
        f"Zbiór testowy: {report.test_legitimate} transakcji prawidłowych, {report.test_fraud} fraudów. "
        f"Próg wyznaczony wyłącznie na walidacji (najniższy próg z FPR < {report.max_false_positive_rate:.0%}) "
        "i zastosowany bez zmian na teście.",
        "",
        f"## Kryterium ze specyfikacji: **{verdict}**",
        "",
        "| Metryka | Autoenkoder | Wymaganie | Status |",
        "|---|---|---|---|",
        f"| Recall (fraud) | {ae.recall:.1%} | ≥ {report.target_recall:.0%} | {_mark(report.meets_recall)} |",
        f"| FPR | {ae.fpr:.2%} | < {report.max_false_positive_rate:.0%} | {_mark(report.meets_fpr)} |",
        "",
        "## Metryki i porównanie z baseline",
        "",
        "| Metryka | Autoenkoder | Isolation Forest |",
        "|---|---|---|",
        *[
            f"| {label} | {formatter(getattr(ae, field))} | {formatter(getattr(base, field))} |"
            for label, field, formatter in (
                ("Recall", "recall", _percent),
                ("FPR", "fpr", _percent2),
                ("Precision", "precision", _percent),
                ("F1", "f1", _decimal),
                ("ROC-AUC", "roc_auc", _decimal),
                ("PR-AUC", "pr_auc", _decimal),
                ("Próg (z walidacji)", "threshold", _decimal),
                ("Recall na walidacji przy tym progu", "validation_recall", _percent),
            )
        ],
        "",
        "Baseline trenowany tak samo jak autoenkoder (tylko transakcje prawidłowe), próg tą samą metodą.",
        "",
        "## Macierze pomyłek (test)",
        "",
        *_confusion_table(ae),
        "",
        *_confusion_table(base),
        "",
        "## Analiza niewykrytych fraudów (false negatives)",
        "",
        f"- Niewykryte: **{fn.missed}** z {fn.missed + fn.detected}; "
        f"z nich {fn.missed_also_by_baseline} nie wykrywa również Isolation Forest.",
        f"- Mediana MSE: niewykryte {_optional(fn.median_mse_missed)}, wykryte {_optional(fn.median_mse_detected)}, "
        f"prawidłowe {fn.median_mse_legitimate:.3f}.",
        "",
        "Najtrudniejsze przypadki (najniższe MSE wśród fraudów):",
        "",
        "| Wiersz testu | MSE | % progu | Percentyl wśród prawidłowych | Wykryty przez baseline |",
        "|---|---|---|---|---|",
        *[
            f"| {case.test_row} | {case.mse:.3f} | {case.share_of_threshold:.0%} | "
            f"{case.legitimate_percentile:.0f} | {'tak' if case.detected_by_baseline else 'nie'} |"
            for case in fn.hardest_cases
        ],
        "",
        "Cechy z największym błędem rekonstrukcji (średni błąd kwadratowy):",
        "",
        "| Cecha | Wykryte fraudy | Niewykryte fraudy | Prawidłowe |",
        "|---|---|---|---|",
        *[
            f"| {error.feature} | {error.detected_fraud:.2f} | {error.missed_fraud:.2f} | {error.legitimate:.2f} |"
            for error in fn.feature_errors
        ],
        "",
        "## Wydajność",
        "",
        f"- Trening autoenkodera: {_seconds(timing.autoencoder_training_seconds)}",
        f"- Predykcja autoenkodera na teście: {timing.autoencoder_test_prediction_seconds:.3f} s",
        f"- Isolation Forest: trening {timing.baseline_fit_seconds:.1f} s, "
        f"predykcja {timing.baseline_test_prediction_seconds:.3f} s",
        f"- Szczytowe zużycie pamięci GPU: {_optional(timing.peak_gpu_memory_mb, '{:.0f} MB')}",
        "",
        "## Model",
        "",
        f"- Warstwy ukryte {model['hidden_dims']}, latent {model['latent_dim']}, aktywacja {model['activation']}, "
        f"batch norm {model['batch_norm']}, dropout {model['dropout']}, parametrów: {report.model_parameters}",
        f"- Optymalizator {report.experiment['optimizer']['name']}, "
        f"lr {report.experiment['optimizer']['learning_rate']:.2e}, "
        f"batch {report.experiment['training']['batch_size']}, "
        f"szum wejścia (denoising) {report.experiment['training']['input_noise_std']}",
        "",
        "## Wykresy",
        "",
        *[f"![{plot.stem}]({plot.name})" for plot in plots],
        "",
    ]
    return "\n".join(lines)


def _confusion_table(result: DetectionResult) -> List[str]:
    return [
        f"**{result.name}**",
        "",
        "| | przewidziane: prawidłowa | przewidziane: fraud |",
        "|---|---|---|",
        f"| **prawidłowa** | TN = {result.tn} | FP = {result.fp} |",
        f"| **fraud** | FN = {result.fn} | TP = {result.tp} |",
    ]


def _mark(passed: bool) -> str:
    return "✅" if passed else "❌"


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _percent2(value: float) -> str:
    return f"{value:.2%}"


def _decimal(value: float) -> str:
    return f"{value:.3f}"


def _seconds(value) -> str:
    return "brak danych" if value is None else f"{value:.0f} s ({value / 60:.1f} min)"


def _optional(value, template: str = "{:.3f}") -> str:
    return "n/a" if value is None else template.format(value)
