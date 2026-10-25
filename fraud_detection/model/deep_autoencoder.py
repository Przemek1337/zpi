from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from fraud_detection.configuration.experiment_config import ModelConfig
from fraud_detection.exceptions import ConfigurationError
from fraud_detection.model.activation_registry import create_activation


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        if config.latent_dim >= input_dim:
            raise ConfigurationError(
                f"latent_dim ({config.latent_dim}) must be smaller than the number of input features ({input_dim})"
            )
        self.input_dim = input_dim
        self.config = config
        encoder_dims = [input_dim, *config.hidden_dims, config.latent_dim]
        decoder_dims = encoder_dims[::-1]
        self.encoder = nn.Sequential(*self._dense_blocks(encoder_dims))
        self.decoder = nn.Sequential(
            *self._dense_blocks(decoder_dims[:-1]),
            nn.Linear(decoder_dims[-2], decoder_dims[-1]),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def _dense_blocks(self, dims: Sequence[int]) -> List[nn.Module]:
        return [self._dense_block(in_dim, out_dim) for in_dim, out_dim in zip(dims, dims[1:])]

    def _dense_block(self, in_dim: int, out_dim: int) -> nn.Module:
        layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if self.config.batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(create_activation(self.config.activation))
        if self.config.dropout > 0:
            layers.append(nn.Dropout(self.config.dropout))
        return nn.Sequential(*layers)
