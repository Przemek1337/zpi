import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from fraud_detection.configuration.experiment_config import ModelConfig, TrainingConfig
from fraud_detection.data.transaction_set import TransactionSet
from fraud_detection.evaluation.reconstruction_error_scorer import ReconstructionErrorScorer
from fraud_detection.exceptions import DataValidationError, TrainingError
from fraud_detection.model.deep_autoencoder import DeepAutoencoder
from fraud_detection.training.autoencoder_trainer import AutoencoderTrainer
from fraud_detection.training.early_stopping import EarlyStopping
from fraud_detection.training.reconstruction_monitor import ReconstructionMonitor, ReconstructionSnapshot
from tests.conftest import make_transaction_set

CPU = torch.device("cpu")


class ScriptedMonitor:
    def __init__(self, losses):
        self._losses = iter(losses)
        self.weights_per_epoch = []

    def evaluate(self, model: nn.Module) -> ReconstructionSnapshot:
        self.weights_per_epoch.append(model.decoder[-1].weight.detach().clone())
        return ReconstructionSnapshot(legitimate_mse=next(self._losses))


def build_trainer(model, epochs, patience=50, learning_rate=1e-2):
    return AutoencoderTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        early_stopping=EarlyStopping(patience=patience),
        config=TrainingConfig(epochs=epochs, batch_size=64),
        device=CPU,
    )


def loader_for(transactions: TransactionSet) -> DataLoader:
    return DataLoader(transactions.to_tensor_dataset(), batch_size=64, shuffle=True)


def test_scorer_returns_per_sample_mse():
    model = DeepAutoencoder(10, ModelConfig(batch_norm=False))
    features = make_transaction_set(legitimate=5, fraud=0).features

    errors = ReconstructionErrorScorer(CPU, batch_size=2).score(model, features)
    expected = ((model(torch.from_numpy(features)) - torch.from_numpy(features)) ** 2).mean(dim=1)

    assert errors.shape == (5,)
    np.testing.assert_allclose(errors, expected.detach().numpy(), rtol=1e-5)


def test_trainer_reduces_validation_loss_and_tracks_fraud_error():
    torch.manual_seed(0)
    model = DeepAutoencoder(10, ModelConfig(hidden_dims=(8,), latent_dim=4))
    monitor = ReconstructionMonitor(ReconstructionErrorScorer(CPU, 64), make_transaction_set(128, 16), 0)

    history = build_trainer(model, epochs=15).fit(loader_for(make_transaction_set(512, 0)), monitor)

    assert len(history.records) == 15
    assert history.records[-1].validation_loss < history.records[0].validation_loss
    assert history.best_record.validation_fraud_mse > history.best_record.validation_loss
    assert history.total_seconds > 0


def test_trainer_restores_best_weights_after_early_stop():
    model = DeepAutoencoder(10, ModelConfig(batch_norm=False))
    monitor = ScriptedMonitor([1.0, 0.5, 0.9, 0.8])

    history = build_trainer(model, epochs=10, patience=2).fit(loader_for(make_transaction_set(128, 0)), monitor)

    assert history.stopped_early
    assert history.best_epoch == 2
    assert len(history.records) == 4
    assert torch.equal(model.decoder[-1].weight, monitor.weights_per_epoch[1])


def test_non_finite_loss_raises_training_error():
    corrupted = make_transaction_set(64, 0)
    corrupted.features[0, 0] = np.nan
    model = DeepAutoencoder(10, ModelConfig(batch_norm=False))

    with pytest.raises(TrainingError, match="Non-finite training loss at epoch 1"):
        build_trainer(model, epochs=1).fit(loader_for(corrupted), ScriptedMonitor([1.0]))


def test_empty_loader_raises_training_error():
    model = DeepAutoencoder(10, ModelConfig(batch_norm=False))
    empty_loader = DataLoader(make_transaction_set(0, 0).to_tensor_dataset(), batch_size=64)

    with pytest.raises(TrainingError, match="no batches"):
        build_trainer(model, epochs=1).fit(empty_loader, ScriptedMonitor([1.0]))


def test_monitor_requires_legitimate_validation_rows():
    with pytest.raises(DataValidationError, match="no legitimate"):
        ReconstructionMonitor(ReconstructionErrorScorer(CPU, 8), make_transaction_set(0, 5), 0)
