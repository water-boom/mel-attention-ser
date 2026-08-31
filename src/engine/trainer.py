"""Trainer and EarlyStopping for Standardized 5-Fold Training with Dynamics Tracking."""

import os
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from ..dynamics.entropy_tracker import EntropyTracker
from ..dynamics.head_diversity import HeadDiversityTracker


class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving."""

    def __init__(self, patience: int = 8, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[dict] = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True


class Trainer:
    """Standardized PyTorch Trainer for SER with Dynamics Hooks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: float = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.early_stopping = early_stopping or EarlyStopping()

        self.entropy_tracker = EntropyTracker()
        self.diversity_tracker = HeadDiversityTracker()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.entropy_tracker.reset_epoch()
        self.diversity_tracker.reset_epoch()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for feats, targets, _ in train_loader:
            feats = feats.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss = self.criterion(logits, targets)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

            # Track dynamics if model supports attention maps
            attn_maps = self.model.get_attention_maps(feats)
            if attn_maps is not None:
                self.entropy_tracker.update(attn_maps)
                self.diversity_tracker.update(attn_maps)

        epoch_loss = total_loss / len(train_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss

        self.entropy_tracker.end_epoch(epoch)
        self.diversity_tracker.end_epoch(epoch)

        return epoch_loss, metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for feats, targets, _ in val_loader:
            feats = feats.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(feats)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = total_loss / len(val_loader.dataset)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss
        return epoch_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 30,
        verbose: bool = True,
    ) -> Dict:
        """Run full training lifecycle with early stopping."""
        history = {
            "train_loss": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_macro_f1": [],
            "val_war": [],
            "val_uar": [],
            "entropy_traj": [],
            "diversity_traj": [],
        }

        best_val_metrics = None

        for epoch in range(1, max_epochs + 1):
            train_loss, train_m = self.train_epoch(train_loader, epoch)
            val_loss, val_m = self.validate(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_macro_f1"].append(train_m["macro_f1"])
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_m["macro_f1"])
            history["val_war"].append(val_m["war"])
            history["val_uar"].append(val_m["uar"])

            improved = self.early_stopping(val_m["macro_f1"], self.model)
            if improved:
                best_val_metrics = val_m

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{max_epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} F1: {train_m['macro_f1']:.4f} | "
                    f"Val Loss: {val_loss:.4f} F1: {val_m['macro_f1']:.4f} WAR: {val_m['war']:.4f}"
                    f"{' [BEST]' if improved else ''}"
                )

            if self.early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

        # Restore best model weights
        if self.early_stopping.best_state_dict is not None:
            self.model.load_state_dict(self.early_stopping.best_state_dict)

        history["entropy_traj"] = self.entropy_tracker.get_trajectory()
        history["diversity_traj"] = self.diversity_tracker.get_trajectory()
        history["best_val_metrics"] = best_val_metrics
        return history
