"""AFCNet model based on the primary definition in notebook cell 11.

Implementation note: the notebook's SpatialAttention.forward overwrote the input
feature map and returned a one-channel tensor. This version preserves the input
feature map and multiplies it by the spatial attention map, as required for a
shape-preserving CBAM module. All other primary Cell 11 module dimensions and
connections are retained.
"""
from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False), nn.ReLU(),
            nn.Linear(hidden, in_channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)
        max_ = self.max_pool(x).flatten(1)
        attn = self.sigmoid(self.fc(avg) + self.fc(max_)).unsqueeze(-1).unsqueeze(-1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_map

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))

class GlobalLocalFusion(nn.Module):
    """Channel-wise gate computed from global average and maximum pooling."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels), nn.ReLU(),
            nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3))
        max_, _ = torch.max(x, dim=2)
        max_, _ = torch.max(max_, dim=2)
        gate = self.fc(torch.cat([avg, max_], dim=1)).unsqueeze(-1).unsqueeze(-1)
        return x * gate

class RDRH(nn.Module):
    """Residual Dense Regression Head from the notebook's primary definition."""
    def __init__(self, in_features: int = 960):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.residual_proj = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.relu(self.fc1(x))
        out = self.dropout(residual)
        out = self.relu(self.fc2(out)) + self.residual_proj(residual) * 0.3
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        return self.fc4(out)

class AFCNet(nn.Module):
    """Return one scalar vehicle-count estimate per image."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        self.features = mobilenet.features
        self.cbam1 = CBAM(40)
        self.cbam2 = CBAM(112)
        self.cbam3 = CBAM(960)
        self.glf = GlobalLocalFusion(960)
        self.regressor = RDRH(960)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 4:
                x = self.cbam1(x)
            elif i == 12:
                x = self.cbam2(x)
        x = self.cbam3(x)
        x = self.glf(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.regressor(x).squeeze(1)
