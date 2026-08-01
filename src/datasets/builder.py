from pathlib import Path

from src.datasets.dataset import build_dataset
from src.datasets.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from torch.utils.data import DataLoader


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_dataloaders(config):
    """
    Build train, validation and test dataloaders.
    """

    train_dataset, classes, class_to_idx = build_dataset(
        Path(config.dataset.train_dir),
        get_train_transforms(config.dataset.image_size),
    )

    val_dataset, _, _ = build_dataset(
        Path(config.dataset.val_dir),
        get_val_transforms(config.dataset.image_size),
    )

    test_dataset, _, _ = build_dataset(
        Path(config.dataset.test_dir),
        get_test_transforms(config.dataset.image_size),
    )

    train_loader = build_dataloader(
        train_dataset,
        config.training.batch_size,
        True,
        config.dataset.num_workers,
    )

    val_loader = build_dataloader(
        val_dataset,
        config.training.batch_size,
        False,
        config.dataset.num_workers,
    )

    test_loader = build_dataloader(
        test_dataset,
        config.training.batch_size,
        False,
        config.dataset.num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        classes,
        class_to_idx,
    )