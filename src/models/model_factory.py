from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.utils.exceptions import ModelError


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    model_name = model_name.lower()

    # Map names to their respective torchvision factory and weights classes
    model_registry = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    }

    if model_name not in model_registry:
        raise ModelError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {', '.join(model_registry.keys())}"
        )

    builder_fn, weights_cls = model_registry[model_name]
    weights = weights_cls.DEFAULT if pretrained else None
    model = builder_fn(weights=weights)

    # Automatically adapt the final classification layer based on the architecture family
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
