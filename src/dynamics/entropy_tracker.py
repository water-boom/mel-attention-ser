"""Attention Entropy Tracking across Training Epochs.

Quantifies the transition of attention distributions from diffuse (high entropy)
to sharp/focused (low entropy) keyframe selection.
"""

import math
from typing import Dict, List, Tuple
import numpy as np
import torch


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute raw and normalized Shannon entropy of attention weights.

    Args:
        attn_weights: Tensor of shape (B, num_heads, T) with sum(dim=-1) == 1
    Returns:
        raw_entropy: (B, num_heads) in nats
        norm_entropy: (B, num_heads) in [0.0, 1.0] relative to uniform distribution
    """
    t_frames = attn_weights.shape[-1]
    max_entropy = math.log(max(t_frames, 1))

    # H(p) = -sum(p * log(p + eps))
    log_p = torch.log(torch.clamp(attn_weights, min=eps))
    raw_entropy = -(attn_weights * log_p).sum(dim=-1)  # (B, num_heads)
    norm_entropy = raw_entropy / (max_entropy + eps)
    return raw_entropy, norm_entropy


class EntropyTracker:
    """Tracks mean attention entropy per epoch during training or validation."""

    def __init__(self):
        self.epoch_history: List[Dict[str, float]] = []
        self._current_batch_entropies: List[float] = []

    def reset_epoch(self):
        self._current_batch_entropies = []

    def update(self, attn_weights: torch.Tensor):
        """Record batch attention weights."""
        if attn_weights is None:
            return
        with torch.no_grad():
            _, norm_ent = compute_attention_entropy(attn_weights)
            mean_val = norm_ent.mean().item()
            self._current_batch_entropies.append(mean_val)

    def end_epoch(self, epoch: int) -> float:
        """Calculate and store mean entropy for the completed epoch."""
        if not self._current_batch_entropies:
            mean_ent = 1.0
        else:
            mean_ent = float(np.mean(self._current_batch_entropies))

        self.epoch_history.append({
            "epoch": epoch,
            "normalized_entropy": mean_ent,
        })
        return mean_ent

    def get_trajectory(self) -> List[Dict[str, float]]:
        return self.epoch_history
