"""Siamese network with a shared ResNet18 image encoder."""
from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class SiameseResNet18(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(backbone.fc.in_features, embedding_dim)
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image).flatten(1)
        return nn.functional.normalize(self.projection(features), p=2, dim=1)
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(image1), self.encode(image2)
