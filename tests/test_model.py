import torch

from src.models.model_factory import build_model


def test_build_model():
    """
    Test model creation.
    """

    model = build_model(
        model_name="resnet18",
        num_classes=10,
        pretrained=False,
    )

    assert isinstance(
        model,
        torch.nn.Module,
    )
