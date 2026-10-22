import json

import pytest
import torch

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.exceptions import DataValidationError
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.training.training_runner import TrainingRunner
from tests.conftest import FEATURE_NAMES, make_transactions


def test_runner_trains_on_csv_and_persists_artifacts(experiment_dict, caplog):
    config = ExperimentConfig.model_validate(experiment_dict)

    history = TrainingRunner(config).run()

    store = ModelArtifactStore(config.output.directory)
    model, feature_names = store.load_model(torch.device("cpu"))
    saved_history = json.loads((store.directory / ModelArtifactStore.HISTORY_FILE).read_text(encoding="utf-8"))
    assert len(history.records) == 3
    assert len(saved_history["records"]) == 3
    assert feature_names == tuple(FEATURE_NAMES[1:])
    assert model(torch.zeros(2, len(feature_names))).shape == (2, len(feature_names))
    assert (store.directory / ModelArtifactStore.CONFIG_FILE).exists()
    assert "Removed 5 non-legitimate transactions" in caplog.text


def test_runner_rejects_mismatched_feature_columns(experiment_dict, write_csv):
    frame = make_transactions(legitimate=20, fraud=2).drop(columns=["V3"])
    experiment_dict["data"]["validation_path"] = str(write_csv("other.csv", frame))

    with pytest.raises(DataValidationError, match=r"missing in validation \['V3'\]"):
        TrainingRunner(ExperimentConfig.model_validate(experiment_dict)).run()


def test_runner_rejects_training_set_without_legitimate_rows(experiment_dict, write_csv):
    experiment_dict["data"]["train_path"] = str(write_csv("frauds.csv", make_transactions(legitimate=0, fraud=10)))

    with pytest.raises(DataValidationError, match="no legitimate transactions"):
        TrainingRunner(ExperimentConfig.model_validate(experiment_dict)).run()
