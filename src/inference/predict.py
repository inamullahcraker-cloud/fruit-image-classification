from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

from src.datasets.transforms import get_test_transforms
from src.models.model_factory import build_model


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    class_mapping_path: str | Path,
    model_name: str,
    image_size: int,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> dict:

    mapping_path = Path(class_mapping_path)

    with mapping_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        class_to_idx = json.load(f)
    """
    Predict the class of a single image.

    Parameters
    ----------
    image_path : str | Path
        Path to the input image.
    checkpoint_path : str | Path
        Path to the trained model checkpoint.
    model_name : str
        Model architecture name.
    image_size : int
        Input image size.
    device : torch.device
        CPU or CUDA device.
    confidence_threshold : float, default=0.5
        Minimum confidence required.

    Returns
    -------
    dict
        Prediction results.
    """

    # --------------------------------------------------
    # Load class names
    # -------------------------------------------------

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = build_model(
        model_name=model_name,
        num_classes=len(idx_to_class),
        pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # --------------------------------------------------
    # Transform image
    # --------------------------------------------------
    transform = get_test_transforms(image_size)

    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    with torch.no_grad():

        outputs = model(image)

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

    # --------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------
    if confidence < confidence_threshold:
        predicted_class = "Unknown"
    else:
        predicted_class = idx_to_class[prediction]

    # --------------------------------------------------
    # Return prediction
    # --------------------------------------------------
    return {
        "class": predicted_class,
        "class_index": prediction,
        "confidence": round(confidence * 100, 2),
    }