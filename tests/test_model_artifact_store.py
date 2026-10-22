import json

import pytest
import torch
import yaml

from fraud_detection.configuration.experiment_config import ExperimentConfig, ModelConfig
from fraud_detection.exceptions import ArtifactError
from fraud_detection.model.deep_autoencoder import DeepAutoencoder
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.training.training_history import EpochRecord, TrainingHistory

CPU = torch.device("cpu")


def test_model_round_trip_preserves_weights_and_features(tmp_path):
    store = ModelArtifactStore(tmp_path)
    model = DeepAutoencoder(10, ModelConfig(hidden_dims=(8,), latent_dim=4))
    store.save_model(model, [f"f{i}" for i in range(10)])

    restored, feature_names = store.load_model(CPU)

    inputs = torch.randn(3, 10)
    model.eval()
    assert torch.allclose(model(inputs), restored(inputs))
    assert feature_names == tuple(f"f{i}" for i in range(10))
    assert restored.config == model.config


def test_history_and_config_are_written(tmp_path):
    store = ModelArtifactStore(tmp_path)
    history = TrainingHistory()
    history.append(EpochRecord(epoch=1, train_loss=1.0, validation_loss=0.9, learning_rate=1e-3, duration_seconds=0.1))
    config = ExperimentConfig.model_validate({"data": {"train_path": "a.csv", "validation_path": "b.csv"}})

    history_path = store.save_history(history)
    config_path = store.save_config(config)

    assert json.loads(history_path.read_text(encoding="utf-8"))["records"][0]["validation_loss"] == 0.9
    assert ExperimentConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8"))) == config


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(ArtifactError, match="not found"):
        ModelArtifactStore(tmp_path).load_model(CPU)


def test_corrupted_checkpoint_is_reported(tmp_path):
    (tmp_path / ModelArtifactStore.MODEL_FILE).write_bytes(b"not a checkpoint")

    with pytest.raises(ArtifactError, match="Failed to read"):
        ModelArtifactStore(tmp_path).load_model(CPU)


def test_checkpoint_with_invalid_metadata_is_reported(tmp_path):
    torch.save({"metadata": {"input_dim": -1}, "state_dict": {}}, tmp_path / ModelArtifactStore.MODEL_FILE)

    with pytest.raises(ArtifactError, match="invalid metadata"):
        ModelArtifactStore(tmp_path).load_model(CPU)
