from __future__ import annotations

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> dict[str, float]:

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is None:

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        else:

            with autocast():

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }
