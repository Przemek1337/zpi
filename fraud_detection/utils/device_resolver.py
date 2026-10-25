import torch

from fraud_detection.exceptions import DeviceUnavailableError


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda":
        _ensure_cuda_device_exists(device)
    return device


def _ensure_cuda_device_exists(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise DeviceUnavailableError(f"Device '{device}' requested but CUDA is not available; use 'cpu' or 'auto'")
    index = device.index or 0
    if index >= torch.cuda.device_count():
        raise DeviceUnavailableError(
            f"Device '{device}' requested but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )
