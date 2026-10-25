from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

ActivationName = Literal["relu", "leaky_relu", "elu", "gelu", "selu", "tanh"]
OptimizerName = Literal["adam", "adamw", "sgd", "rmsprop"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(FrozenConfig):
    level: LogLevel = "INFO"
    file: Optional[Path] = None


class DataConfig(FrozenConfig):
    # either a splits archive produced by the preprocessing notebook, or a pair of CSV files
    splits_path: Optional[Path] = None
    preparation_notebook: Optional[Path] = None
    train_path: Optional[Path] = None
    validation_path: Optional[Path] = None
    # CSV mode only: held-out test file, read exclusively by the one-off test evaluation
    test_path: Optional[Path] = None
    label_column: str = Field("Class", min_length=1)
    legitimate_label: int = 0
    drop_columns: Tuple[str, ...] = ()

    @property
    def uses_splits_archive(self) -> bool:
        return self.splits_path is not None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DataConfig:
        has_csv = self.train_path is not None or self.validation_path is not None
        if self.uses_splits_archive == has_csv:
            raise ValueError("set either splits_path or both train_path and validation_path")
        if has_csv and (self.train_path is None or self.validation_path is None):
            raise ValueError("train_path and validation_path must be set together")
        if self.preparation_notebook is not None and not self.uses_splits_archive:
            raise ValueError("preparation_notebook requires splits_path")
        if self.test_path is not None and self.uses_splits_archive:
            raise ValueError("test_path is for CSV sources; the splits archive already contains the test split")
        return self

    @model_validator(mode="after")
    def _label_column_is_kept(self) -> DataConfig:
        if self.label_column in self.drop_columns:
            raise ValueError(f"label column '{self.label_column}' cannot be listed in drop_columns")
        return self


class ModelConfig(FrozenConfig):
    hidden_dims: Tuple[PositiveInt, ...] = (24, 16)
    latent_dim: PositiveInt = 8
    activation: ActivationName = "leaky_relu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    batch_norm: bool = True

    @model_validator(mode="after")
    def _latent_is_bottleneck(self) -> ModelConfig:
        if self.hidden_dims and self.latent_dim >= min(self.hidden_dims):
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be smaller than every hidden layer {list(self.hidden_dims)}"
            )
        return self


class OptimizerConfig(FrozenConfig):
    name: OptimizerName = "adam"
    learning_rate: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 0.0


class SchedulerConfig(FrozenConfig):
    enabled: bool = True
    factor: float = Field(0.5, gt=0.0, lt=1.0)
    patience: NonNegativeInt = 3
    min_learning_rate: NonNegativeFloat = 1e-6


class TrainingConfig(FrozenConfig):
    epochs: PositiveInt = 100
    batch_size: PositiveInt = 256
    gradient_clip_norm: Optional[PositiveFloat] = 1.0
    num_workers: NonNegativeInt = 0
    # denoising: std of Gaussian noise added to training inputs; the target stays the clean transaction
    input_noise_std: NonNegativeFloat = 0.0


class EarlyStoppingConfig(FrozenConfig):
    enabled: bool = True
    patience: PositiveInt = 10
    min_delta: NonNegativeFloat = 0.0


class EvaluationConfig(FrozenConfig):
    # fraud recall is reported at the threshold that keeps validation FPR strictly below this budget
    max_false_positive_rate: float = Field(0.01, gt=0.0, lt=1.0)


class OutputConfig(FrozenConfig):
    directory: Path = Path("artifacts/autoencoder")


class ExperimentConfig(FrozenConfig):
    seed: NonNegativeInt = 42
    device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
