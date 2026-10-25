from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Type

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fraud_detection.configuration.experiment_config import OptimizerConfig, SchedulerConfig
from fraud_detection.exceptions import ConfigurationError

_OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}

SUPPORTED_OPTIMIZERS: FrozenSet[str] = frozenset(_OPTIMIZERS)


def create_optimizer(parameters: Iterable[nn.Parameter], config: OptimizerConfig) -> optim.Optimizer:
    if config.name not in _OPTIMIZERS:
        raise ConfigurationError(f"Unknown optimizer '{config.name}', supported: {sorted(SUPPORTED_OPTIMIZERS)}")
    return _OPTIMIZERS[config.name](parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def create_scheduler(optimizer: optim.Optimizer, config: SchedulerConfig) -> Optional[ReduceLROnPlateau]:
    if not config.enabled:
        return None
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor,
        patience=config.patience,
        min_lr=config.min_learning_rate,
    )
