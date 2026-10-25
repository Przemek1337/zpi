import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluate_autoencoder import ExitCode, main
from fraud_detection.configuration.experiment_config import DataConfig, ExperimentConfig
from fraud_detection.data.dataset_source import load_test
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.test_evaluation import ScoredSplits, evaluate_detector
from fraud_detection.exceptions import DataLoadingError
from fraud_detection.training.training_runner import TrainingRunner
from tests.conftest import make_transactions, save_splits


def labelled(labels):
    labels = np.array(labels, dtype=np.int64)
    return TransactionSet(features=np.zeros((len(labels), 1), dtype=np.float32), labels=labels, feature_names=("x",))


@pytest.fixture
def trained_model(tmp_path) -> Path:
    splits = save_splits(
        tmp_path / "splits.npz",
        make_transactions(legitimate=400, fraud=0, seed=1),
        make_transactions(legitimate=200, fraud=12, seed=2),
        make_transactions(legitimate=200, fraud=12, seed=3),
    )
    config = ExperimentConfig.model_validate({
        "seed": 3,
        "device": "cpu",
        "data": {"splits_path": str(splits), "drop_columns": ["Time"]},
        "model": {"hidden_dims": [16], "latent_dim": 4},
        "training": {"epochs": 3, "batch_size": 64},
        "output": {"directory": str(tmp_path / "model")},
    })
    TrainingRunner(config).run()
    return config.output.directory


def test_detector_threshold_comes_from_validation_only():
    validation = labelled([0] * 100 + [1] * 4)
    test = labelled([0] * 10 + [1] * 4)
    scores = ScoredSplits(
        validation=np.concatenate([np.arange(100.0), [50.0, 150.0, 160.0, 170.0]]),
        # test legit scores far above validation ones: a test-tuned threshold would hide these false positives
        test=np.array([200.0] * 2 + [1.0] * 8 + [300.0, 300.0, 50.0, 10.0]),
    )

    result = evaluate_detector("detector", scores, validation, test, legitimate_label=0, max_fpr=0.01)

    assert result.threshold == 99.0
    assert result.validation_recall == 0.75
    assert (result.tp, result.fn, result.fp, result.tn) == (2, 2, 2, 8)
    assert result.recall == 0.5
    assert result.fpr == 0.2
    assert result.precision == 0.5


def test_test_split_is_read_from_splits_archive(tmp_path):
    splits = save_splits(
        tmp_path / "splits.npz", make_transactions(10, 0), make_transactions(5, 1), make_transactions(7, 3)
    )

    test = load_test(DataConfig(splits_path=splits))

    assert test.class_counts() == {0: 7, 1: 3}


def test_csv_source_requires_test_path(write_csv):
    config = DataConfig(train_path=write_csv("a.csv", make_transactions(5, 0)), validation_path="b.csv")

    with pytest.raises(DataLoadingError, match="test_path"):
        load_test(config)


def test_evaluation_writes_report_plots_and_verdict(trained_model, caplog):
    assert main(["--model-dir", str(trained_model)]) == ExitCode.SUCCESS

    output = trained_model / "test_evaluation"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert report["test_legitimate"] == 200 and report["test_fraud"] == 12
    autoencoder = report["autoencoder"]
    assert autoencoder["tp"] + autoencoder["fn"] == 12
    assert autoencoder["fp"] + autoencoder["tn"] == 200
    assert report["baseline"]["name"] == "Isolation Forest"
    assert report["criteria"]["recall_met"] == (autoencoder["recall"] >= 0.85)
    assert len(report["false_negatives"]["hardest_cases"]) == 10
    for plot in ("mse_distribution", "confusion_matrices", "precision_recall", "learning_curves"):
        assert (output / f"{plot}.png").is_file()
        assert f"({plot}.png)" in markdown
    assert "Kryterium ze specyfikacji" in markdown
    assert "TEST EVALUATION" in (output / "evaluation.log").read_text(encoding="utf-8")


def test_repeated_evaluation_warns(trained_model, capsys):
    main(["--model-dir", str(trained_model)])
    capsys.readouterr()

    main(["--model-dir", str(trained_model)])

    assert "already holds a test evaluation" in capsys.readouterr().err


def test_custom_output_directory(trained_model, tmp_path):
    assert main(["--model-dir", str(trained_model), "--output", str(tmp_path / "report")]) == ExitCode.SUCCESS
    assert (tmp_path / "report" / "report.md").is_file()


def test_missing_model_directory_is_a_known_error(tmp_path, capsys):
    assert main(["--model-dir", str(tmp_path / "missing")]) == ExitCode.KNOWN_ERROR
    assert "ConfigurationError" in capsys.readouterr().err


def test_feature_mismatch_is_reported(trained_model, capsys):
    config_path = trained_model / "experiment_config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["data"]["drop_columns"] = ["V1"]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert main(["--model-dir", str(trained_model)]) == ExitCode.KNOWN_ERROR
    assert "differ from the model" in capsys.readouterr().err
