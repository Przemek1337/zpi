import pytest
import torch

from fraud_detection.exceptions import DeviceUnavailableError
from fraud_detection.utils.device_resolver import resolve_device


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_cuda_without_gpu_is_reported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
        resolve_device("cuda")


def test_nonexistent_cuda_index_is_reported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(DeviceUnavailableError, match="1 CUDA device"):
        resolve_device("cuda:3")
