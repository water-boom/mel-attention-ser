"""Attention Blocks Zoo for 2D Spectro-Temporal and 1D Sequence Processing."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-Excitation Channel Attention Block.

    Compresses (F, T) spatial dimensions to 1x1 scalar per channel.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced_dim = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, F, T)
        Returns:
            Tensor of shape (B, C, F, T)
        """
        b, c, _, _ = x.shape
        # Global Average Pooling over F and T
        z = x.mean(dim=(2, 3))  # (B, C)
        s = self.fc(z).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * s


class CoordinateAttention(nn.Module):
    """Coordinate Attention for Spectro-Temporal Representations.

    Decouples horizontal (Time) and vertical (Frequency) dimensions via 1D strip pooling.
    Preserves exact frequency coordinates and time dynamics.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced_dim = max(in_channels // reduction, 8)

        self.conv1 = nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced_dim)
        self.act = nn.ReLU(inplace=True)

        self.conv_f = nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, F, T)
        Returns:
            Tensor of shape (B, C, F, T)
        """
        b, c, f, t = x.shape

        # 1. Frequency pool (pool along T) -> (B, C, F, 1)
        z_f = x.mean(dim=3, keepdim=True)
        # 2. Time pool (pool along F) -> (B, C, 1, T)
        z_t = x.mean(dim=2, keepdim=True)

        # 3. Concatenate along spatial dimension after transposing time
        # z_t.permute(0, 1, 3, 2) is (B, C, T, 1)
        # z_cat is (B, C, F + T, 1)
        z_cat = torch.cat([z_f, z_t.permute(0, 1, 3, 2)], dim=2)

        # 4. Joint encoding
        y = self.act(self.bn1(self.conv1(z_cat)))

        # 5. Split back into frequency and time parts
        y_f, y_t = torch.split(y, [f, t], dim=2)
        y_t = y_t.permute(0, 1, 3, 2)  # back to (B, C, 1, T)

        # 6. Generate attention weights via Sigmoid
        attn_f = torch.sigmoid(self.conv_f(y_f))  # (B, C, F, 1)
        attn_t = torch.sigmoid(self.conv_t(y_t))  # (B, C, 1, T)

        return x * attn_f * attn_t


class MultiHeadAttentivePooling(nn.Module):
    """Multi-Head Attentive Pooling (MHAP) for sequence frame aggregation.

    Splits representation into H independent attention heads to capture diverse acoustic cues.
    """

    def __init__(self, in_dim: int, num_heads: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.w_proj = nn.Linear(in_dim, hidden_dim * num_heads)
        self.v_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(in_dim * num_heads, in_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Sequence tensor of shape (B, T, D)
        Returns:
            pooled: (B, D) aggregated representation
            attn_weights: (B, num_heads, T) attention distributions
        """
        b, t, d = h.shape

        # (B, T, H * hidden_dim) -> (B, T, H, hidden_dim)
        u = torch.tanh(self.w_proj(h)).view(b, t, self.num_heads, self.hidden_dim)

        # Score per head: (B, T, H, 1) -> (B, H, T)
        scores = self.v_proj(u).squeeze(-1).permute(0, 2, 1)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T)

        # Weighted sum: (B, H, 1, T) @ (B, 1, T, D) -> (B, H, D)
        h_expanded = h.unsqueeze(1)  # (B, 1, T, D)
        attn_expanded = attn_weights.unsqueeze(2)  # (B, H, 1, T)
        head_contexts = torch.matmul(attn_expanded, h_expanded).squeeze(2)  # (B, H, D)

        # Concatenate heads and project back
        concat_contexts = head_contexts.view(b, self.num_heads * d)
        pooled = self.out_proj(concat_contexts)

        return pooled, attn_weights


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP).

    Computes attention-weighted mean (mu) and standard deviation (sigma) to capture
    both 1st-order central tendency and 2nd-order dynamic fluctuation.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.score_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.out_proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Sequence tensor of shape (B, T, D)
        Returns:
            pooled: (B, D) aggregated representation with 1st & 2nd order statistics
            attn_weights: (B, 1, T) attention distribution
        """
        # (B, T, 1) -> (B, 1, T)
        scores = self.score_net(h).squeeze(-1).unsqueeze(1)
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, T)

        # Weighted mean: mu = sum(alpha_t * h_t)
        mu = torch.matmul(attn_weights, h)  # (B, 1, D)

        # Weighted variance: sigma^2 = sum(alpha_t * (h_t - mu)^2)
        var = torch.matmul(attn_weights, (h - mu) ** 2)  # (B, 1, D)
        sigma = torch.sqrt(torch.clamp(var, min=self.eps))  # (B, 1, D)

        # Concatenate [mu, sigma] and project
        stats = torch.cat([mu.squeeze(1), sigma.squeeze(1)], dim=-1)  # (B, 2D)
        pooled = self.out_proj(stats)  # (B, D)

        return pooled, attn_weights
