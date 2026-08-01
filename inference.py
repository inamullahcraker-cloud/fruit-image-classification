"""
Main inference script for the Fruit Image Classification project.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config.config import load_config
from src.inference.predict import predict_image
from src.utils.device import get_device
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def load_class_mapping(
    path: str | Path = "artifacts/class_to_idx.json",
) -> list[str]:
    """
    Load class names from the saved class mapping.
    """

    mapping_path = Path(path)

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {mapping_path}"
        )

    with mapping_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        class_to_idx = json.load(file)

    return list(class_to_idx.keys())


def main() -> None:
    """
    Run single-image inference.
    """

    config = load_config(
        "configs/config.yaml",
    )

    device = get_device(
        config.training.device,
    )

    classes = load_class_mapping()

    image_path = "data/test/apple/apple_01.jpg"

    checkpoint_path = (
        "artifacts/checkpoints/best_model.pth"
    )

    prediction = predict_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        class_names=classes,
        model_name=config.model.name,
        image_size=config.dataset.image_size,
        device=device,
    )

    LOGGER.info("Prediction Results")
    LOGGER.info("----------------------------")

    for key, value in prediction.items():
        LOGGER.info("%s : %s", key, value)


if __name__ == "__main__":
    main()