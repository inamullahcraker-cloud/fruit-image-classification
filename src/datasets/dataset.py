from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class FruitDataset(Dataset):
    """
    Custom dataset for fruit image classification.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform=None,
    ) -> None:

        if len(image_paths) != len(labels):
            raise ValueError(
                "Number of images and labels must match."
            )

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ):
        image_path = self.image_paths[index]

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(
                image=image
            )["image"]

        return image, torch.tensor(
            label,
            dtype=torch.long,
        )
from pathlib import Path


def build_dataset(
    root_dir: Path,
    transform=None,
):
    """
    Build a FruitDataset from a directory.

    Directory structure:

    root/
        apple/
        banana/
        grape/
    """

    image_paths = []
    labels = []

    classes = sorted(
        [
            d.name
            for d in root_dir.iterdir()
            if d.is_dir()
        ]
    )

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    for class_name in classes:

        class_dir = root_dir / class_name

        for image_path in class_dir.iterdir():

            if image_path.suffix.lower() in valid_extensions:

                image_paths.append(image_path)

                labels.append(
                    class_to_idx[class_name]
                )

    dataset = FruitDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform,
    )

    return dataset, classes, class_to_idx