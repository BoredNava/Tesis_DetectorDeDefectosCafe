"""Compact image classifier built from depthwise-separable convolution blocks."""
from __future__ import annotations
import torch
from torch import nn

class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class SlimCNN(nn.Module):
    """Small binary classifier; channel widths and dropout come from YAML."""
    def __init__(self, num_classes: int = 2, channels: list[int] | None = None, dropout: float = 0.25):
        super().__init__()
        channels = channels or [24, 48, 96, 160]
        blocks=[]; in_channels=3
        for index, out_channels in enumerate(channels):
            blocks.append(SeparableBlock(in_channels, out_channels, stride=1 if index == 0 else 2))
            in_channels=out_channels
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_channels, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
