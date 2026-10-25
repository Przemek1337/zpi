from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import TensorDataset


class TransactionSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    features: np.ndarray
    labels: np.ndarray
    feature_names: Tuple[str, ...]

    @model_validator(mode="after")
    def _shapes_are_consistent(self) -> TransactionSet:
        if self.features.ndim != 2:
            raise ValueError(f"features must be a 2D array, got {self.features.ndim}D")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be a 1D array, got {self.labels.ndim}D")
        if len(self.features) != len(self.labels):
            raise ValueError(f"features have {len(self.features)} rows but labels have {len(self.labels)}")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features have {self.features.shape[1]} columns but {len(self.feature_names)} names were given"
            )
        return self

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def feature_count(self) -> int:
        return self.features.shape[1]

    def class_counts(self) -> Dict[int, int]:
        classes, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(classes.tolist(), counts.tolist()))

    def with_label(self, label: int) -> TransactionSet:
        return self._select(self.labels == label)

    def excluding_label(self, label: int) -> TransactionSet:
        return self._select(self.labels != label)

    def to_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(torch.from_numpy(self.features))

    def _select(self, mask: np.ndarray) -> TransactionSet:
        return TransactionSet(features=self.features[mask], labels=self.labels[mask], feature_names=self.feature_names)
