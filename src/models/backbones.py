"""Unified 4-Layer 2D-CNN Backbone and Attention Model Architectures."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SELayer,
    CoordinateAttention,
    MultiHeadAttentivePooling,
    AttentiveStatisticsPooling,
)


class BaseSERModel(nn.Module, ABC):
    """Abstract Base Class for Speech Emotion Recognition Models."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (B, num_classes)."""
        pass

    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional method to extract attention distributions for probing and visualization.

        Returns:
            Tensor of shape (B, num_heads, T) or None
        """
        return None


class ConvBlock(nn.Module):
    """Standard 2D Convolutional Block with optional In-Backbone Attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_stride: Tuple[int, int] = (2, 2),
        attention_type: Optional[str] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=pool_stride)

        if attention_type == "se":
            self.attn = SELayer(out_channels)
        elif attention_type == "coord":
            self.attn = CoordinateAttention(out_channels)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        if self.attn is not None:
            x = self.attn(x)
        x = self.pool(x)
        return x


class CNNBase(BaseSERModel):
    """Baseline: 4-Layer 2D-CNN with Global Average Pooling (GAP)."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__(num_classes)
        self.layer1 = ConvBlock(in_channels, 32, (2, 2))
        self.layer2 = ConvBlock(32, 64, (2, 2))
        self.layer3 = ConvBlock(64, 128, (2, 2))
        self.layer4 = ConvBlock(128, 256, (2, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.gap(x).flatten(1)
        logits = self.classifier(feat)
        return logits


class CNN_SE(BaseSERModel):
    """Channel Attention: 4-Layer 2D-CNN with SELayer inside each conv block."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__(num_classes)
        self.layer1 = ConvBlock(in_channels, 32, (2, 2), attention_type="se")
        self.layer2 = ConvBlock(32, 64, (2, 2), attention_type="se")
        self.layer3 = ConvBlock(64, 128, (2, 2), attention_type="se")
        self.layer4 = ConvBlock(128, 256, (2, 1), attention_type="se")

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.gap(x).flatten(1)
        logits = self.classifier(feat)
        return logits


class CNN_Coord(BaseSERModel):
    """Spectro-Temporal Coordinate Attention: Preserves 1D Time & Frequency Coordinates."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__(num_classes)
        self.layer1 = ConvBlock(in_channels, 32, (2, 2), attention_type=None)
        self.layer2 = ConvBlock(32, 64, (2, 2), attention_type="coord")
        self.layer3 = ConvBlock(64, 128, (2, 2), attention_type="coord")
        self.layer4 = ConvBlock(128, 256, (2, 1), attention_type="coord")

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.gap(x).flatten(1)
        logits = self.classifier(feat)
        return logits


class CNN_MHAP(BaseSERModel):
    """Temporal Multi-Head Attentive Pooling: Dynamic keyframe selection at terminal pooling."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3, num_heads: int = 4):
        super().__init__(num_classes)
        self.layer1 = ConvBlock(in_channels, 32, (2, 2))
        self.layer2 = ConvBlock(32, 64, (2, 2))
        self.layer3 = ConvBlock(64, 128, (2, 2))
        self.layer4 = ConvBlock(128, 256, (2, 1))

        # (B, C=256, F=8, T) -> project C*F to 256
        self.time_proj = nn.Linear(256 * 8, 256)
        self.mhap = MultiHeadAttentivePooling(in_dim=256, num_heads=num_heads, hidden_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _extract_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 256, 8, T)

        # Transpose to (B, T, C * F)
        b, c, f, t = x.shape
        x_seq = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h = F.relu(self.time_proj(x_seq))  # (B, T, 256)
        return h, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._extract_sequence(x)
        pooled, _ = self.mhap(h)
        logits = self.classifier(pooled)
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._extract_sequence(x)
        _, attn_weights = self.mhap(h)
        return attn_weights  # (B, num_heads, T)


class CNN_ASP(BaseSERModel):
    """Attentive Statistics Pooling: Captures 1st-order mean and 2nd-order dynamic fluctuation."""

    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__(num_classes)
        self.layer1 = ConvBlock(in_channels, 32, (2, 2))
        self.layer2 = ConvBlock(32, 64, (2, 2))
        self.layer3 = ConvBlock(64, 128, (2, 2))
        self.layer4 = ConvBlock(128, 256, (2, 1))

        self.time_proj = nn.Linear(256 * 8, 256)
        self.asp = AttentiveStatisticsPooling(in_dim=256, hidden_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _extract_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, f, t = x.shape
        x_seq = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h = F.relu(self.time_proj(x_seq))
        return h, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._extract_sequence(x)
        pooled, _ = self.asp(h)
        logits = self.classifier(pooled)
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self._extract_sequence(x)
        _, attn_weights = self.asp(h)
        return attn_weights  # (B, 1, T)


class Wav2Vec2_MHAP(BaseSERModel):
    """Optional Anchor: Frozen wav2vec2-base SSL Representation + MHAP Head."""

    def __init__(self, num_classes: int = 8, num_heads: int = 4, in_dim: int = 768):
        super().__init__(num_classes)
        self.in_dim = in_dim
        self.mhap = MultiHeadAttentivePooling(in_dim=in_dim, num_heads=num_heads, hidden_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes pre-extracted wav2vec2 sequence features of shape (B, T, 768)."""
        pooled, _ = self.mhap(x)
        return self.classifier(pooled)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        _, attn_weights = self.mhap(x)
        return attn_weights
