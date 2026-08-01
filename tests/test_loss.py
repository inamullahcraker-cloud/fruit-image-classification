import torch.nn as nn

from src.losses.loss_factory import build_loss


def test_build_loss():

    criterion = build_loss(
        "cross_entropy",
    )

    assert isinstance(
        criterion,
        nn.Module,
    )
