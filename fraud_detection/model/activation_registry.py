from typing import Dict, FrozenSet, Type

from torch import nn

from fraud_detection.exceptions import ConfigurationError

_ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}

SUPPORTED_ACTIVATIONS: FrozenSet[str] = frozenset(_ACTIVATIONS)


def create_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ConfigurationError(f"Unknown activation '{name}', supported: {sorted(SUPPORTED_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()
