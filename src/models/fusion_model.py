"""
Dual-Stream Spectral Fusion Model for produce shelf-life intelligence.
Integrates RGB visual surface patterns with NIR subsurface spectral signals.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from src.models.rgb_model import RGBProduceModel
from src.models.spectral_model import NIRSpectralModel


class SpecViTCrossAttentionFusion(nn.Module):
    """
    SpecViT Cross-Attention Fusion Block (BioVision Proposed Architecture).
    Cross-attends high-resolution RGB visual tokens with 700-1050nm Near-Infrared
    spectral tokens to model sub-surface biochemical decay.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout_p: float = 0.25):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, rgb_feat: torch.Tensor, nir_feat: torch.Tensor) -> torch.Tensor:
        q = rgb_feat.unsqueeze(1)
        k = nir_feat.unsqueeze(1)
        v = nir_feat.unsqueeze(1)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        x = self.norm1(q + attn_out)
        out = self.norm2(x + self.ffn(x))
        return out.squeeze(1)


class GatedMultimodalFusion(nn.Module):
    """
    Gated fusion mechanism allowing dynamic weighting between visual surface
    cues and internal spectral indicators based on specimen condition.
    """

    def __init__(self, rgb_dim: int = 256, nir_dim: int = 256, out_dim: int = 256, dropout_p: float = 0.25):
        super().__init__()
        in_dim = rgb_dim + nir_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, rgb_feat: torch.Tensor, nir_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([rgb_feat, nir_feat], dim=-1)
        gated = combined * self.gate(combined)
        fused = self.projector(gated)
        return fused


class DualStreamSpectralFusionModel(nn.Module):
    """
    BioVision Dual-Stream SpecViT Architecture.
    Simultaneously processes RGB visual data and NIR spectral reflectance
    using SpecViT cross-attention to predict ripeness stage, shelf life, and detect latent subsurface spoilage.
    """

    def __init__(
        self,
        num_classes: int = 5,
        feature_dim: int = 256,
        dropout_p: float = 0.25,
        rgb_pretrained: bool = True,
        fusion_type: str = "specvit",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type

        # Stream 1: RGB Backbone
        self.rgb_stream = RGBProduceModel(
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout_p=dropout_p,
            pretrained=rgb_pretrained,
        )

        # Stream 2: NIR Spectral Backbone
        self.nir_stream = NIRSpectralModel(
            in_channels=1,
            num_classes=num_classes,
            feature_dim=feature_dim,
            dropout_p=dropout_p,
        )

        # Multimodal SpecViT Cross-Attention or Gated Fusion Layer
        if fusion_type == "specvit":
            self.fusion = SpecViTCrossAttentionFusion(
                embed_dim=feature_dim,
                num_heads=4,
                dropout_p=dropout_p,
            )
        else:
            self.fusion = GatedMultimodalFusion(
                rgb_dim=feature_dim,
                nir_dim=feature_dim,
                out_dim=feature_dim,
                dropout_p=dropout_p,
            )

        # Multi-task Prediction Heads
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

        self.shelf_life_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1),
        )

        self.degradation_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.latent_defect_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual streams and fusion head.

        Args:
            rgb: (B, 3, 224, 224)
            nir: (B, 1, 224, 224)
            return_features: if True, returns intermediate representations
        """
        rgb_feat = self.rgb_stream.extract_features(rgb)
        nir_feat = self.nir_stream.extract_features(nir)

        fused_feat = self.fusion(rgb_feat, nir_feat)

        logits = self.classifier(fused_feat)
        shelf_life = self.shelf_life_head(fused_feat).squeeze(-1)
        degradation = self.degradation_head(fused_feat).squeeze(-1)
        latent_defect_prob = self.latent_defect_head(fused_feat).squeeze(-1)

        result = {
            "logits": logits,
            "shelf_life_days": shelf_life,
            "degradation_score": degradation,
            "latent_defect_prob": latent_defect_prob,
        }

        if return_features:
            result["rgb_feat"] = rgb_feat
            result["nir_feat"] = nir_feat
            result["fused_feat"] = fused_feat

        return result
