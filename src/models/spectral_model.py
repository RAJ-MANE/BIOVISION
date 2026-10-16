"""
NIR Spectral produce freshness model.
Custom deep convolutional network specialized for subsurface reflectance,
water absorption gradients, and internal cellular tissue degradation.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with batch norm, activation, and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        out = self.pool(out)
        return out


class NIRSpectralModel(nn.Module):
    """
    Lightweight, high-sensitivity NIR spectral network.
    Extracts deep subsurface structural and reflectance features from near-infrared bands.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        feature_dim: int = 256,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 4 Residual Conv stages: 1 -> 32 -> 64 -> 128 -> 256
        self.stage1 = ConvBlock(in_channels, 32, pool=True)     # 224 -> 112
        self.stage2 = ConvBlock(32, 64, pool=True)              # 112 -> 56
        self.stage3 = ConvBlock(64, 128, pool=True)             # 56 -> 28
        self.stage4 = ConvBlock(128, 256, pool=True)            # 28 -> 14

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # Standalone prediction heads (for NIR-alone ablation)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.shelf_life_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dimensional spectral embedding vector."""
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        pooled = self.avgpool(out)
        flattened = torch.flatten(pooled, 1)
        embedding = self.projector(flattened)
        return embedding

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.extract_features(x)
        logits = self.classifier(emb)
        shelf_life = self.shelf_life_head(emb).squeeze(-1)
        degradation = self.degradation_head(emb).squeeze(-1)

        return {
            "logits": logits,
            "shelf_life_days": shelf_life,
            "degradation_score": degradation,
            "features": emb,
        }
