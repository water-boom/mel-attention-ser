"""Multi-Head Diversity and Specialization Tracker.

Quantifies whether multiple attention heads learn diverse, complementary acoustic cues
(high diversity/orthogonality) or collapse into redundant copies (low diversity).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


def compute_head_orthogonality(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean pairwise cosine divergence across attention heads.

    Args:
        attn_weights: Tensor of shape (B, num_heads, T)
    Returns:
        Tensor of shape (B,) with mean pairwise diversity in [0.0, 1.0]
    """
    b, num_heads, t = attn_weights.shape
    if num_heads <= 1:
        return torch.zeros(b, device=attn_weights.device)

    # Normalize each head vector: (B, num_heads, T)
    norm = torch.norm(attn_weights, p=2, dim=-1, keepdim=True) + eps
    unit_heads = attn_weights / norm

    # Pairwise cosine similarity: (B, num_heads, num_heads)
    cosine_sim = torch.bmm(unit_heads, unit_heads.transpose(1, 2))

    # Mask out diagonal (self-similarity)
    mask = 1.0 - torch.eye(num_heads, device=attn_weights.device).unsqueeze(0)
    num_pairs = num_heads * (num_heads - 1)

    mean_pairwise_sim = (cosine_sim * mask).sum(dim=(1, 2)) / num_pairs
    pairwise_diversity = 1.0 - mean_pairwise_sim
    return torch.clamp(pairwise_diversity, min=0.0, max=1.0)


class HeadDiversityTracker:
    """Tracks multi-head specialization index across training epochs."""

    def __init__(self):
        self.epoch_history: List[Dict[str, float]] = []
        self._current_batch_divs: List[float] = []

    def reset_epoch(self):
        self._current_batch_divs = []

    def update(self, attn_weights: torch.Tensor):
        if attn_weights is None or attn_weights.shape[1] <= 1:
            return
        with torch.no_grad():
            div = compute_head_orthogonality(attn_weights)
            self._current_batch_divs.append(div.mean().item())

    def end_epoch(self, epoch: int) -> float:
        if not self._current_batch_divs:
            mean_div = 0.0
        else:
            mean_div = float(np.mean(self._current_batch_divs))

        self.epoch_history.append({
            "epoch": epoch,
            "head_diversity": mean_div,
        })
        return mean_div

    def get_trajectory(self) -> List[Dict[str, float]]:
        return self.epoch_history
