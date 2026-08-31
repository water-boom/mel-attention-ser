"""Training, optimization, and metrics engine."""
from .metrics import compute_metrics, classification_summary
from .trainer import Trainer, EarlyStopping

__all__ = ["compute_metrics", "classification_summary", "Trainer", "EarlyStopping"]
