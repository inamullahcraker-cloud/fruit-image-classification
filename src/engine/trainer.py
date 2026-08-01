from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

from src.engine.train import train_one_epoch
from src.engine.validate import validate_one_epoch
from src.utils.logger import get_logger


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        scheduler=None,
        scaler: GradScaler | None = None,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler

        self.logger = get_logger(__name__)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_loss = float("inf")
        self.history = []

    def fit(
        self,
        epochs: int,
    ):

        for epoch in range(epochs):

            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scaler,
            )

            val_metrics = validate_one_epoch(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )

            self.logger.info(
                "Epoch %d/%d | Train Loss: %.4f | "
                "Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            if val_metrics["loss"] < self.best_loss:

                self.best_loss = val_metrics["loss"]

                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / "best_model.pth",
                )

                self.logger.info("Best model saved.")

        return self.history
