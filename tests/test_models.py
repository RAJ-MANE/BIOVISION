"""
Unit tests for BioVision neural network architectures and uncertainty estimators.
"""
import pytest
import torch

from src.models.rgb_model import RGBProduceModel
from src.models.spectral_model import NIRSpectralModel
from src.models.fusion_model import DualStreamSpectralFusionModel
from src.models.uncertainty import MonteCarloUncertaintyEstimator


def test_rgb_model_forward():
    model = RGBProduceModel(num_classes=5, feature_dim=256, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert "logits" in out
    assert out["logits"].shape == (2, 5)
    assert "shelf_life_days" in out
    assert out["shelf_life_days"].shape == (2,)
    assert "degradation_score" in out
    assert out["degradation_score"].shape == (2,)
    assert "features" in out
    assert out["features"].shape == (2, 256)


def test_nir_spectral_model_forward():
    model = NIRSpectralModel(in_channels=1, num_classes=5, feature_dim=256)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)

    assert "logits" in out
    assert out["logits"].shape == (2, 5)
    assert "shelf_life_days" in out
    assert out["shelf_life_days"].shape == (2,)
    assert "degradation_score" in out
    assert out["degradation_score"].shape == (2,)


def test_fusion_model_forward():
    model = DualStreamSpectralFusionModel(num_classes=5, feature_dim=256, rgb_pretrained=False)
    rgb = torch.randn(2, 3, 224, 224)
    nir = torch.randn(2, 1, 224, 224)

    out = model(rgb, nir, return_features=True)

    assert out["logits"].shape == (2, 5)
    assert out["shelf_life_days"].shape == (2,)
    assert out["degradation_score"].shape == (2,)
    assert out["latent_defect_prob"].shape == (2,)
    assert out["fused_feat"].shape == (2, 256)


def test_mc_dropout_uncertainty_estimator():
    model = DualStreamSpectralFusionModel(num_classes=5, feature_dim=256, rgb_pretrained=False)
    estimator = MonteCarloUncertaintyEstimator(model=model, num_samples=5)

    rgb = torch.randn(1, 3, 224, 224)
    nir = torch.randn(1, 1, 224, 224)

    results = estimator.estimate(rgb, nir)

    assert "predicted_class_idx" in results
    assert 0 <= results["predicted_class_idx"] < 5
    assert len(results["class_probabilities"]) == 5
    assert "mean_shelf_life_days" in results
    assert "ci_95_shelf_life" in results
    assert len(results["ci_95_shelf_life"]) == 2
    assert results["ci_95_shelf_life"][0] <= results["ci_95_shelf_life"][1]
    assert results["reliability"] in ["HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", "HIGH_UNCERTAINTY"]
