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


def test_monitor_reports_recall_at_false_positive_budget():
    class FixedScorer:
        def score(self, model, features):
            return features[:, 0]

    validation = TransactionSet(
        features=np.array([[float(value)] for value in [*range(100), 50.5, 98.5, 150.0, 200.0]], dtype=np.float32),
        labels=np.array([0] * 100 + [1] * 4, dtype=np.int64),
        feature_names=("score",),
    )

    snapshot = ReconstructionMonitor(FixedScorer(), validation, 0, max_false_positive_rate=0.01).evaluate(None)

    # threshold is the largest legitimate score (99), so only 150 and 200 are detected
    assert snapshot.fraud_recall == 0.5
    assert snapshot.legitimate_mse == pytest.approx(49.5)


def test_monitor_without_fraud_has_no_recall():
    monitor = ReconstructionMonitor(ReconstructionErrorScorer(CPU, 8), make_transaction_set(20, 0), 0)

    snapshot = monitor.evaluate(DeepAutoencoder(10, ModelConfig(batch_norm=False)))

    assert snapshot.fraud_mse is None
    assert snapshot.fraud_recall is None


def test_trainer_records_validation_recall_and_reports_each_epoch():
    torch.manual_seed(0)
    model = DeepAutoencoder(10, ModelConfig(hidden_dims=(8,), latent_dim=4))
    monitor = ReconstructionMonitor(ReconstructionErrorScorer(CPU, 64), make_transaction_set(200, 20), 0)
    seen = []

    history = build_trainer(model, epochs=3).fit(loader_for(make_transaction_set(256, 0)), monitor, seen.append)

    assert [record.epoch for record in seen] == [1, 2, 3]
    assert all(0.0 <= record.validation_fraud_recall <= 1.0 for record in history.records)


def test_epoch_callback_can_abort_training():
    class Abort(Exception):
        pass

    def abort_after_second_epoch(record):
        if record.epoch == 2:
            raise Abort

    model = DeepAutoencoder(10, ModelConfig(batch_norm=False))

    with pytest.raises(Abort):
        build_trainer(model, epochs=10).fit(
            loader_for(make_transaction_set(64, 0)), ScriptedMonitor([1.0] * 10), abort_after_second_epoch
        )


def test_denoising_feeds_noisy_inputs_but_targets_clean_ones():
    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.decoder = nn.Sequential(nn.Linear(1, 1))
            self.inputs = []

        def forward(self, inputs):
            self.inputs.append(inputs.detach().clone())
            return inputs * self.weight

    model = RecordingModel()
    clean = make_transaction_set(64, 0)
    trainer = AutoencoderTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        early_stopping=EarlyStopping(patience=5),
        config=TrainingConfig(epochs=1, batch_size=64, input_noise_std=0.5),
        device=CPU,
    )

    history = trainer.fit(DataLoader(clean.to_tensor_dataset(), batch_size=64), ScriptedMonitor([1.0]))

    noise = model.inputs[0] - torch.from_numpy(clean.features)
    assert noise.std().item() == pytest.approx(0.5, rel=0.1)
    # identity model: loss is exactly the injected noise measured against the clean target
    assert history.records[0].train_loss == pytest.approx((noise ** 2).mean().item(), rel=1e-5)
