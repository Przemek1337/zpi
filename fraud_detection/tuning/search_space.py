from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import optuna
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.exceptions import ConfigurationError
from fraud_detection.tuning.tuning_config import Choice, FloatRange, IntRange, SearchSpaceConfig
from fraud_detection.utils.validation_messages import describe_validation_error

Overrides = Dict[str, Dict[str, Any]]


class SearchSpace:
    """Maps an Optuna trial onto experiment config overrides.

    The same mapping serves sampling during the search and rebuilding the winner from its stored
    parameters (``optuna.trial.FixedTrial``), so both always produce the identical configuration.
    """

    def __init__(self, config: SearchSpaceConfig) -> None:
        self._space = config

    def suggest(self, trial: optuna.trial.BaseTrial) -> Overrides:
        latent_dim = self._int(trial, "latent_dim", self._space.latent_dim)
        hidden_dims = self._hidden_dims(trial, latent_dim)
        denoising = self._choice(trial, "denoising", self._space.denoising)
        scheduler_enabled = self._choice(trial, "scheduler_enabled", self._space.scheduler_enabled)
        return {
            "model": {
                "hidden_dims": list(hidden_dims),
                "latent_dim": latent_dim,
                "activation": self._choice(trial, "activation", self._space.activation),
                "dropout": self._float(trial, "dropout", self._space.dropout),
                "batch_norm": self._choice(trial, "batch_norm", self._space.batch_norm),
            },
            "optimizer": {
                "name": self._choice(trial, "optimizer", self._space.optimizer),
                "learning_rate": self._float(trial, "learning_rate", self._space.learning_rate),
                "weight_decay": self._float(trial, "weight_decay", self._space.weight_decay),
            },
            "training": {
                "batch_size": self._choice(trial, "batch_size", self._space.batch_size),
                "gradient_clip_norm": self._choice(trial, "gradient_clip_norm", self._space.gradient_clip_norm),
                "input_noise_std": (
                    self._float(trial, "input_noise_std", self._space.input_noise_std) if denoising else 0.0
                ),
            },
            "scheduler": {
                "enabled": scheduler_enabled,
                **(
                    {
                        "factor": self._float(trial, "scheduler_factor", self._space.scheduler_factor),
                        "patience": self._int(trial, "scheduler_patience", self._space.scheduler_patience),
                    }
                    if scheduler_enabled
                    else {}
                ),
            },
        }

    def _hidden_dims(self, trial: optuna.trial.BaseTrial, latent_dim: int) -> Tuple[int, ...]:
        layers = self._int(trial, "hidden_layers", self._space.hidden_layers)
        first = self._choice(trial, "first_hidden_dim", self._space.first_hidden_dim)
        shrink = self._float(trial, "width_shrink", self._space.width_shrink) if layers > 1 else 1.0
        # every hidden layer must stay wider than the bottleneck
        return tuple(max(round(first * shrink**index), latent_dim + 1) for index in range(layers))

    @staticmethod
    def _int(trial: optuna.trial.BaseTrial, name: str, spec: IntRange) -> int:
        return trial.suggest_int(name, spec.low, spec.high, log=spec.log)

    @staticmethod
    def _float(trial: optuna.trial.BaseTrial, name: str, spec: FloatRange) -> float:
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)

    @staticmethod
    def _choice(trial: optuna.trial.BaseTrial, name: str, spec: Choice) -> Any:
        return trial.suggest_categorical(name, list(spec.choices))


def apply_overrides(base: ExperimentConfig, *override_sets: Mapping[str, Mapping[str, Any]]) -> ExperimentConfig:
    """Returns ``base`` with the given sections partially replaced, validated like a config file."""
    raw = base.model_dump(mode="json")
    for overrides in override_sets:
        for section, values in overrides.items():
            raw[section] = {**raw.get(section, {}), **values} if isinstance(values, Mapping) else values
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"Sampled configuration is invalid:\n{describe_validation_error(error)}") from error
