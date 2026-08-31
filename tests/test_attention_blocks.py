"""Unit tests for Attention Blocks and Unified Backbone Architectures."""

import pytest
import torch
from src.models.blocks import (
    SELayer,
    CoordinateAttention,
    MultiHeadAttentivePooling,
    AttentiveStatisticsPooling,
)
from src.models.registry import build_model, list_models


def test_se_layer():
    block = SELayer(in_channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 75)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_coordinate_attention():
    block = CoordinateAttention(in_channels=64, reduction=16)
    x = torch.randn(2, 64, 32, 75)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_mhap_pooling():
    block = MultiHeadAttentivePooling(in_dim=256, num_heads=4, hidden_dim=128)
    h = torch.randn(2, 75, 256)
    pooled, attn = block(h)
    assert pooled.shape == (2, 256)
    assert attn.shape == (2, 4, 75)
    # Check Softmax normalization along time
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_asp_pooling():
    block = AttentiveStatisticsPooling(in_dim=256, hidden_dim=128)
    h = torch.randn(2, 75, 256)
    pooled, attn = block(h)
    assert pooled.shape == (2, 256)
    assert attn.shape == (2, 1, 75)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 1), atol=1e-5)


def test_all_models_forward_and_backward():
    model_names = list_models()
    x = torch.randn(2, 3, 128, 300)

    for name in model_names:
        if name == "w2v2_mhap":
            # w2v2 expects (B, T, 768)
            x_in = torch.randn(2, 75, 768)
        else:
            x_in = x

        model = build_model(name, num_classes=8)
        logits = model(x_in)
        assert logits.shape == (2, 8), f"Failed shape check on model {name}"

        # Test backprop gradient
        loss = logits.sum()
        loss.backward()

        for param_name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing gradient for {param_name} in {name}"
