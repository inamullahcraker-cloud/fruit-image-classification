"""
Training curve visualization utilities.

This module provides functions for plotting and saving
training and validation loss/accuracy curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    history: list[dict],
    save_dir: str | Path = "reports/figures",
) -> dict[str, Path]:
    """
    Plot and save training history curves.

    Parameters
    ----------
    history : list[dict]
        Training history returned by the Trainer.
    save_dir : str | Path, default="reports/figures"
        Directory where the figures will be saved.

    Returns
    -------
    dict[str, Path]
        Dictionary containing paths to the saved figures.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = [item["epoch"] for item in history]

    train_loss = [
        item["train_loss"]
        for item in history
    ]

    val_loss = [
        item["val_loss"]
        for item in history
    ]

    train_acc = [
        item["train_accuracy"]
        for item in history
    ]

    val_acc = [
        item["val_accuracy"]
        for item in history
    ]

    # --------------------------------------------------
    # Loss Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    loss_path = save_dir / "loss_curve.png"

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Accuracy Curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_acc,
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    accuracy_path = save_dir / "accuracy_curve.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "loss_curve": loss_path,
        "accuracy_curve": accuracy_path,
    }