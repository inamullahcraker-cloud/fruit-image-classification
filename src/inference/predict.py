"""
Predictor class for image inference.

Loads the model and class mapping once and reuses them for
multiple predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

from src.datasets.transforms import get_test_transforms
from src.models.model_factory import build_model


class Predictor:
    """
    Loads a trained model once and performs predictions.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        class_mapping_path: str | Path,
        model_name: str,
        image_size: int,
        device: torch.device,
        confidence_threshold: float = 0.5,
    ) -> None:

        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # -----------------------------
        # Load class mapping
        # -----------------------------

        mapping_path = Path(class_mapping_path)

        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Class mapping not found: {mapping_path}"
            )

        with mapping_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.class_to_idx = json.load(file)

        self.idx_to_class = {
            v: k for k, v in self.class_to_idx.items()
        }

        # -----------------------------
        # Build model
        # -----------------------------

        self.model = build_model(
            model_name=model_name,
            num_classes=len(self.idx_to_class),
            pretrained=False,
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )

        self.model.to(device)
        self.model.eval()

        # -----------------------------
        # Create transform once
        # -----------------------------

        self.transform = get_test_transforms(
            image_size
        )

    def predict(
        self,
        image_path: str | Path,
    ) -> dict:

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = self.transform(
            image=image
        )["image"]

        image = image.unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        confidence = confidence.item()
        prediction = prediction.item()

        if confidence < self.confidence_threshold:

            predicted_class = "Unknown"

        else:

            predicted_class = self.idx_to_class.get(
                prediction,
                "Unknown",
            )

        return {
            "prediction": predicted_class,
            "class_index": prediction,
            "confidence": round(confidence, 4),
        }