"""Learning dynamics probes for attention entropy and multi-head diversity."""
from .entropy_tracker import EntropyTracker, compute_attention_entropy
from .head_diversity import HeadDiversityTracker, compute_head_orthogonality

__all__ = [
    "EntropyTracker",
    "compute_attention_entropy",
    "HeadDiversityTracker",
    "compute_head_orthogonality",
]
