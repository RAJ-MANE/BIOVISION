"""
RGB produce freshness model based on MobileNetV3-Small.
Extracts surface color, visible skin texture, and external defects.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class RGBProduceModel(nn.Module):
    """
    Lightweight RGB vision model for produce grading and surface defect detection.
    Can operate standalone as a baseline, or provide visual features to the fusion model.
    """

    def __init__(
        self,
        num_classes: int = 5,
        feature_dim: int = 256,
        dropout_p: float = 0.25,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Backbone: MobileNetV3-Small
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
        except Exception:
            base_model = models.mobilenet_v3_small(weights=None)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3-Small has 576 output channels before classifier
        in_features = 576
        self.projector = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        # Standalone prediction heads (for RGB-alone ablation)
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
        """Extract 256-dimensional visual embedding vector."""
        feats = self.features(x)
        pooled = self.avgpool(feats)
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
