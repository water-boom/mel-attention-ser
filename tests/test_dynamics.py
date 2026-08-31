"""Unit tests for Learning Dynamics: Entropy & Head Diversity."""

import math
import pytest
import torch
from src.dynamics.entropy_tracker import EntropyTracker, compute_attention_entropy
from src.dynamics.head_diversity import HeadDiversityTracker, compute_head_orthogonality


def test_attention_entropy():
    # 1. Uniform distribution on 100 frames
    t = 100
    uniform_weights = torch.ones(2, 4, t) / t
    raw_ent, norm_ent = compute_attention_entropy(uniform_weights)
    # Uniform normalized entropy should be ~1.0
    assert torch.allclose(norm_ent, torch.ones_like(norm_ent), atol=1e-3)

    # 2. Dirac delta / single frame focus
    delta_weights = torch.zeros(2, 4, t)
    delta_weights[:, :, 0] = 1.0
    raw_ent, norm_ent = compute_attention_entropy(delta_weights)
    # Focused normalized entropy should be ~0.0
    assert torch.allclose(norm_ent, torch.zeros_like(norm_ent), atol=1e-3)


def test_head_orthogonality():
    # 1. Identical heads (collapse)
    t = 50
    head = torch.randn(2, 1, t).softmax(dim=-1)
    collapsed_heads = head.repeat(1, 4, 1)
    div_collapsed = compute_head_orthogonality(collapsed_heads)
    # Diversity should be 0.0
    assert torch.allclose(div_collapsed, torch.zeros_like(div_collapsed), atol=1e-4)

    # 2. Disjoint / orthogonal heads
    disjoint_heads = torch.zeros(2, 4, 40)
    disjoint_heads[:, 0, 0:10] = 0.1
    disjoint_heads[:, 1, 10:20] = 0.1
    disjoint_heads[:, 2, 20:30] = 0.1
    disjoint_heads[:, 3, 30:40] = 0.1
    div_disjoint = compute_head_orthogonality(disjoint_heads)
    # Diversity should be 1.0
    assert torch.allclose(div_disjoint, torch.ones_like(div_disjoint), atol=1e-4)
