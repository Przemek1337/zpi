import pytest
import torch
from torch import nn

from fraud_detection.configuration.experiment_config import ModelConfig
from fraud_detection.exceptions import ConfigurationError
from fraud_detection.model.activation_registry import create_activation
from fraud_detection.model.deep_autoencoder import DeepAutoencoder


def test_reconstruction_has_input_shape():
    model = DeepAutoencoder(30, ModelConfig(hidden_dims=(20, 14), latent_dim=7))

    assert model(torch.randn(16, 30)).shape == (16, 30)


def test_encoder_compresses_to_latent_dim():
    model = DeepAutoencoder(30, ModelConfig(hidden_dims=(20, 14), latent_dim=7))

    assert model.encode(torch.randn(4, 30)).shape == (4, 7)


def test_decoder_output_layer_is_linear():
    model = DeepAutoencoder(30, ModelConfig(dropout=0.2))

    assert isinstance(model.decoder[-1], nn.Linear)
    assert model.parameter_count > 0


def test_optional_layers_follow_config():
    plain = DeepAutoencoder(30, ModelConfig(batch_norm=False, dropout=0.0))
    regularized = DeepAutoencoder(30, ModelConfig(batch_norm=True, dropout=0.3))

    assert not any(isinstance(module, (nn.BatchNorm1d, nn.Dropout)) for module in plain.modules())
    assert any(isinstance(module, nn.BatchNorm1d) for module in regularized.modules())
    assert any(isinstance(module, nn.Dropout) for module in regularized.modules())


def test_latent_not_smaller_than_input_is_rejected():
    with pytest.raises(ConfigurationError, match="latent_dim"):
        DeepAutoencoder(8, ModelConfig(hidden_dims=(16,), latent_dim=8))


def test_unknown_activation_is_rejected():
    with pytest.raises(ConfigurationError, match="activation"):
        create_activation("swish")
