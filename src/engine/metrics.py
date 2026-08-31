"""Evaluation Metrics: Macro-F1, Weighted Accuracy (WAR), Unweighted Average Recall (UAR)."""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> Dict[str, Union[float, np.ndarray]]:
    """Compute standard Speech Emotion Recognition metrics.

    Args:
        y_true: Array of true class labels (0 to num_classes-1)
        y_pred: Array of predicted class labels
    Returns:
        Dict with 'macro_f1', 'war' (accuracy), 'uar' (macro recall), and 'confusion_matrix'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    war = float(accuracy_score(y_true, y_pred))
    uar = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": macro_f1,
        "war": war,
        "uar": uar,
        "confusion_matrix": cm,
    }


def classification_summary(metrics: Dict[str, Union[float, np.ndarray]]) -> str:
    """Format metrics dictionary into a clean string for logging."""
    return (
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"WAR (Acc): {metrics['war']:.4f} | "
        f"UAR (Rec): {metrics['uar']:.4f}"
    )
