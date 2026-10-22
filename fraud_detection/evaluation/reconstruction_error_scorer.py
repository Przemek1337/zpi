from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReconstructionErrorScorer:
    def __init__(self, device: torch.device, batch_size: int) -> None:
        self._device = device
        self._batch_size = batch_size

    def score(self, model: nn.Module, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        model.eval()
        errors = []
        with torch.no_grad():
            for (batch,) in self._batches(features):
                batch = batch.to(self._device)
                errors.append(((model(batch) - batch) ** 2).mean(dim=1).cpu())
        return torch.cat(errors).numpy()

    def _batches(self, features: np.ndarray) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(features))
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
