"""
Main inference script for the Fruit Image Classification project.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config.config import load_config
from src.inference.predict import Predictor
from src.utils.device import get_device
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)



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

   

    image_path = "data/test/apple/apple_01.jpg"

    

    predictions= Predictor(
        checkpoint_path=config.inference.checkpoint_path,
        class_mapping_path=config.inference.class_mapping,
        model_name=config.model.name,
        image_size=config.dataset.image_size,
        device=device,
    )
    prediction=predictions.predict(image_path=image_path,)

    LOGGER.info("Prediction Results")
    LOGGER.info("----------------------------")

    for key, value in prediction.items():
        LOGGER.info("%s : %s", key, value)


if __name__ == "__main__":
    main()