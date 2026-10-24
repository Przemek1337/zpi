from __future__ import annotations

from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from fraud_detection.configuration.experiment_config import ActivationName, FrozenConfig, OptimizerName

ChoiceT = TypeVar("ChoiceT")


def _ensure_range(low: float, high: float, log: bool) -> None:
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    if log and low <= 0:
        raise ValueError(f"log-scaled range needs a positive low bound, got {low}")


class IntRange(FrozenConfig):
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> IntRange:
        _ensure_range(self.low, self.high, self.log)
        return self


class FloatRange(FrozenConfig):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> FloatRange:
        _ensure_range(self.low, self.high, self.log)
        return self


class Choice(FrozenConfig, Generic[ChoiceT]):
    choices: Tuple[ChoiceT, ...] = Field(min_length=1)


class SearchSpaceConfig(FrozenConfig):
    """Ranges sampled for every trial; a single-element choice (or low == high) pins a hyperparameter."""

    # architecture: encoder widths shrink geometrically from first_hidden_dim towards the bottleneck
    hidden_layers: IntRange = IntRange(low=1, high=4)
    first_hidden_dim: Choice[PositiveInt] = Choice(choices=(16, 24, 32, 48, 64, 96, 128))
    width_shrink: FloatRange = FloatRange(low=0.3, high=1.0)
    latent_dim: IntRange = IntRange(low=2, high=16)
    activation: Choice[ActivationName] = Choice(choices=("relu", "leaky_relu", "elu", "gelu", "selu", "tanh"))
    dropout: FloatRange = FloatRange(low=0.0, high=0.3)
    batch_norm: Choice[bool] = Choice(choices=(True, False))
    # denoising: Gaussian noise added to training inputs only; the reconstruction target stays clean
    denoising: Choice[bool] = Choice(choices=(True, False))
    input_noise_std: FloatRange = FloatRange(low=0.01, high=0.5, log=True)
    optimizer: Choice[OptimizerName] = Choice(choices=("adam", "adamw", "sgd", "rmsprop"))
    learning_rate: FloatRange = FloatRange(low=1e-5, high=1e-2, log=True)
    weight_decay: FloatRange = FloatRange(low=1e-8, high=1e-3, log=True)
    batch_size: Choice[PositiveInt] = Choice(choices=(128, 256, 512, 1024, 2048))
    gradient_clip_norm: Choice[Optional[PositiveFloat]] = Choice(choices=(None, 0.5, 1.0, 5.0))
    scheduler_enabled: Choice[bool] = Choice(choices=(True, False))
    scheduler_factor: FloatRange = FloatRange(low=0.1, high=0.8)
    scheduler_patience: IntRange = IntRange(low=1, high=8)

    @model_validator(mode="after")
    def _sizes_are_positive(self) -> SearchSpaceConfig:
        if self.hidden_layers.low < 1 or self.latent_dim.low < 1:
            raise ValueError("hidden_layers and latent_dim must be at least 1")
        if not 0.0 < self.width_shrink.low <= self.width_shrink.high <= 1.0:
            raise ValueError("width_shrink must lie in (0, 1]")
        if self.dropout.low < 0.0 or self.dropout.high >= 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learning_rate.low <= 0 or self.input_noise_std.low < 0 or self.weight_decay.low < 0:
            raise ValueError("learning_rate must be positive; input_noise_std and weight_decay non-negative")
        if not 0.0 < self.scheduler_factor.low <= self.scheduler_factor.high < 1.0:
            raise ValueError("scheduler_factor must lie in (0, 1)")
        if self.scheduler_patience.low < 0:
            raise ValueError("scheduler_patience must be non-negative")
        return self


class StudyConfig(FrozenConfig):
    name: str = Field("autoencoder-recall-at-fpr", min_length=1)
    # journal file shared by all workers; re-running with the same file resumes the study
    storage: Path = Path("artifacts/tuning/study.journal")
    sampler_seed: int = 42
    # trials sampled at random before TPE starts modelling the search space
    random_startup_trials: NonNegativeInt = 20


class BudgetConfig(FrozenConfig):
    timeout_hours: PositiveFloat = 9.0
    max_trials: Optional[PositiveInt] = None
    parallel_workers: PositiveInt = 3


class TrialTrainingConfig(FrozenConfig):
    # epoch budget per trial; the final model is trained with the base config's (longer) budget
    epochs: PositiveInt = 60
    early_stopping_patience: PositiveInt = 8


class PruningConfig(FrozenConfig):
    enabled: bool = True
    startup_trials: NonNegativeInt = 10
    warmup_epochs: NonNegativeInt = 10


class PreflightConfig(FrozenConfig):
    require_gpu: bool = True
    smoke_trials: PositiveInt = 3
    smoke_epochs: PositiveInt = 2
    smoke_train_rows: PositiveInt = 4096
    smoke_validation_legitimate_rows: PositiveInt = 4096
    worker_start_timeout_seconds: PositiveInt = 600


class FinalTrainingConfig(FrozenConfig):
    enabled: bool = True


class TuningOutputConfig(FrozenConfig):
    directory: Path = Path("artifacts/tuning")
    top_trials: PositiveInt = 10


class TuningConfig(FrozenConfig):
    # experiment config providing data, seed, device, logging and the final training budget
    base_config: Path = Path("config/autoencoder_training.yaml")
    study: StudyConfig = Field(default_factory=StudyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    trial: TrialTrainingConfig = Field(default_factory=TrialTrainingConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    preflight: PreflightConfig = Field(default_factory=PreflightConfig)
    final_training: FinalTrainingConfig = Field(default_factory=FinalTrainingConfig)
    output: TuningOutputConfig = Field(default_factory=TuningOutputConfig)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
