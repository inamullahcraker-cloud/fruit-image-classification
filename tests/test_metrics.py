import torch

from src.metrics.classification_metrics import (
    compute_classification_metrics
)


def test_accuracy():

    predictions = torch.tensor(
        [0, 1, 2, 3]
    )

    targets = torch.tensor(
        [0, 1, 2, 3]
    )

    metrics = compute_classification_metrics(
    predictions,
    targets,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
