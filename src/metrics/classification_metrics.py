from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Any | None = None,
    average: str = "weighted",
) -> dict[str, Any]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : list[int]
        Ground truth labels.
    y_pred : list[int]
        Predicted labels.
    y_prob : Any | None
        Prediction probabilities (optional).
    average : str
        Averaging strategy for multiclass metrics.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
            )
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics
