from __future__ import annotations

import torch
from tqdm import tqdm


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> dict:

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }
