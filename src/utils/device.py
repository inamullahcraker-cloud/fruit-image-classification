"""
Device management utilities.
"""

from __future__ import annotations

import torch

from src.utils.exceptions import ConfigurationError


def get_device(device_name: str = "auto") -> torch.device:
    """
    Return the appropriate torch device.

    Parameters
    ----------
    device_name : str
        One of: "auto", "cpu", or "cuda".

    Returns
    -------
    torch.device
        Selected device.

    Raises
    ------
    ConfigurationError
        If an invalid or unavailable device is requested.
    """
    device_name = device_name.lower()

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("CUDA requested but not available.")
        return torch.device("cuda")

    raise ConfigurationError(
        f"Unsupported device '{device_name}'. "
        "Choose from: auto, cpu, cuda."
    )
