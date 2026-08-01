"""
Main training script for the Fruit Image Classification project.
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

from src.config.config import load_config
from src.datasets.builder import build_dataloaders
from src.engine.trainer import Trainer
from src.losses.loss_factory import build_loss
from src.models.model_factory import build_model
from src.utils.device import get_device
from src.utils.exceptions import (
    ConfigurationError,
    DatasetError,
    ModelError,
)
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.visualization.training_curves import (
    plot_training_curves,
)

LOGGER = get_logger(__name__)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def save_class_mapping(
    class_to_idx: dict[str, int],
    output_dir: str | Path = "artifacts",
) -> None:
    """Save class-to-index mapping for inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "class_to_idx.json"

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_to_idx,
            file,
            indent=4,
        )


def save_training_history(
    history: list[dict],
    output_dir: str | Path = "artifacts",
) -> None:
    """Save training history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_file = output_dir / "training_history.json"

    with history_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
def main() -> None:
    """Main training pipeline."""
    try:
        LOGGER.info("=" * 60)
        LOGGER.info("Fruit Image Classification Training")
        LOGGER.info("=" * 60)

        # --------------------------------------------------
        # Load Config
        # --------------------------------------------------
        config = load_config("configs/config.yaml")
        LOGGER.info("Configuration loaded successfully.")

        # --------------------------------------------------
        # Set Seed
        # --------------------------------------------------
        set_seed(config.training.seed)
        LOGGER.info("Random seed set to %d", config.training.seed)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------
        device = get_device(config.training.device)
        LOGGER.info("Using device: %s", device)

        # --------------------------------------------------
        # DataLoaders
        # --------------------------------------------------
        (
            train_loader,
            val_loader,
            test_loader,
            classes,
            class_to_idx,
        ) = build_dataloaders(config)

        LOGGER.info("Training samples   : %d", len(train_loader.dataset))
        LOGGER.info("Validation samples : %d", len(val_loader.dataset))
        LOGGER.info("Test samples       : %d", len(test_loader.dataset))
        LOGGER.info("Number of classes  : %d", len(classes))

        save_class_mapping(class_to_idx)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        model = build_model(
            model_name=config.model.name,
            num_classes=len(classes),
            pretrained=config.model.pretrained,
        )
        model.to(device)
        LOGGER.info("Model '%s' created.", config.model.name)

        # --------------------------------------------------
        # Loss Function
        # --------------------------------------------------
        criterion = build_loss(
            loss_name=config.training.loss_name,
            label_smoothing=config.training.label_smoothing,
        )
        LOGGER.info("Loss function: %s", config.training.loss_name)

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        LOGGER.info("Optimizer: Adam")
        LOGGER.info("Learning Rate : %.6f", config.training.learning_rate)
        LOGGER.info("Weight Decay  : %.6f", config.training.weight_decay)

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.training.lr_step_size,
            gamma=config.training.lr_gamma,
        )
        LOGGER.info("Scheduler: StepLR")
        LOGGER.info("Step Size : %d", config.training.lr_step_size)
        LOGGER.info("Gamma     : %.4f", config.training.lr_gamma)

        # --------------------------------------------------
        # Scaler
        # --------------------------------------------------
        scaler = GradScaler(enabled=config.training.use_amp)
        LOGGER.info("Mixed Precision: %s", config.training.use_amp)

        # --------------------------------------------------
        # Trainer
        # --------------------------------------------------
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_dir="artifacts/checkpoints",
        )

        # Optional: Load external class names file if needed
        class_names_path = Path("artifacts/class_names.json")
        if class_names_path.exists():
            with class_names_path.open("r", encoding="utf-8") as f:
                class_names = json.load(f)

        LOGGER.info("Trainer initialized successfully.")

        # --------------------------------------------------
        # Trainer Fit
        # --------------------------------------------------
        LOGGER.info("Starting training for %d epochs...", config.training.epochs)
        history = trainer.fit(epochs=config.training.epochs)
        LOGGER.info("Training completed successfully.")

        # --------------------------------------------------
        # Save History
        # --------------------------------------------------
        save_training_history(
            history=history,
            output_dir="artifacts",
        )
        LOGGER.info("Training history saved.")

        # --------------------------------------------------
        # Plot Curves
        # --------------------------------------------------
        plot_training_curves(
            history=history,
            save_dir="reports/figures",
        )
        LOGGER.info("Training curves generated.")

        LOGGER.info("=" * 60)
        LOGGER.info("Training pipeline finished successfully.")
        LOGGER.info("=" * 60)

    except (
        ConfigurationError,
        DatasetError,
        ModelError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        LOGGER.exception("Training failed: %s", error)
        raise


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main()