from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.exceptions import ConfigurationError


def build_loss(
    loss_name: str = "cross_entropy",
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Build and return a loss function.

    Parameters
    ----------
    loss_name : str
        Name of the loss function.
    class_weights : torch.Tensor | None
        Optional class weights.
    label_smoothing : float
        Label smoothing factor.

    Returns
    -------
    nn.Module
        Configured loss function.
    """

    loss_name = loss_name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    raise ConfigurationError(
        f"Unsupported loss function: {loss_name}"
    )
