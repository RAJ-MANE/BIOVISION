"""
Monte Carlo Dropout Uncertainty Estimation for Produce Freshness Prediction.
Provides calibrated confidence intervals and flags out-of-distribution or ambiguous scans.
"""
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np


class MonteCarloUncertaintyEstimator:
    """
    Performs Monte Carlo Dropout sampling across multiple stochastic forward passes
    to compute predictive mean, variance, epistemic uncertainty, and 95% confidence intervals.
    """

    def __init__(self, model: nn.Module, num_samples: int = 20, uncertainty_threshold: float = 0.35):
        self.model = model
        self.num_samples = num_samples
        self.uncertainty_threshold = uncertainty_threshold

    def _enable_dropout(self):
        """Enable dropout layers while keeping batchnorm in eval mode."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def estimate(
        self,
        rgb_tensor: torch.Tensor,
        nir_tensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Run stochastic forward passes for input tensors.

        Args:
            rgb_tensor: (1, 3, H, W)
            nir_tensor: (1, 1, H, W)
            device: torch device

        Returns:
            Dictionary containing means, stds, confidence intervals, and uncertainty flag.
        """
        if device is None:
            device = next(self.model.parameters()).device

        rgb_tensor = rgb_tensor.to(device)
        nir_tensor = nir_tensor.to(device)

        self._enable_dropout()

        logits_list = []
        shelf_life_list = []
        degradation_list = []
        latent_defect_list = []

        for _ in range(self.num_samples):
            outputs = self.model(rgb_tensor, nir_tensor)
            logits = outputs["logits"].cpu()
            probs = torch.softmax(logits, dim=-1)
            logits_list.append(probs.numpy())

            shelf_life = outputs["shelf_life_days"].cpu().numpy()
            shelf_life_list.append(shelf_life)

            degradation = outputs["degradation_score"].cpu().numpy()
            degradation_list.append(degradation)

            if "latent_defect_prob" in outputs:
                latent_defect_list.append(outputs["latent_defect_prob"].cpu().numpy())

        # Reset model back to standard eval mode
        self.model.eval()

        # Compute statistics across MC passes
        probs_array = np.concatenate(logits_list, axis=0)           # (S, num_classes)
        shelf_life_array = np.concatenate(shelf_life_list, axis=0)  # (S,)
        degradation_array = np.concatenate(degradation_list, axis=0) # (S,)

        mean_probs = np.mean(probs_array, axis=0)
        predicted_class_idx = int(np.argmax(mean_probs))
        
        # Predictive entropy: H(y|x) = -sum(p * log(p))
        eps = 1e-8
        predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
        # Normalized entropy (max is log(5) = ~1.609)
        max_entropy = np.log(mean_probs.shape[0])
        norm_entropy = float(predictive_entropy / max_entropy)

        mean_shelf_life = float(np.mean(shelf_life_array))
        std_shelf_life = float(np.std(shelf_life_array))
        ci_shelf_life = [
            float(np.percentile(shelf_life_array, 2.5)),
            float(np.percentile(shelf_life_array, 97.5)),
        ]

        mean_degradation = float(np.mean(degradation_array))
        std_degradation = float(np.std(degradation_array))

        mean_latent_defect = (
            float(np.mean(np.concatenate(latent_defect_list, axis=0)))
            if latent_defect_list
            else 0.0
        )

        # Evaluate reliability
        is_high_uncertainty = (norm_entropy > self.uncertainty_threshold) or (std_shelf_life > 2.0)
        
        if norm_entropy < 0.20 and std_shelf_life <= 1.0:
            reliability = "HIGH_CONFIDENCE"
            reliability_badge = "success"
        elif norm_entropy <= self.uncertainty_threshold and std_shelf_life <= 2.0:
            reliability = "MODERATE_CONFIDENCE"
            reliability_badge = "warning"
        else:
            reliability = "HIGH_UNCERTAINTY"
            reliability_badge = "danger"

        return {
            "predicted_class_idx": predicted_class_idx,
            "class_probabilities": mean_probs.tolist(),
            "mean_shelf_life_days": max(0.0, round(mean_shelf_life, 2)),
            "std_shelf_life_days": round(std_shelf_life, 2),
            "ci_95_shelf_life": [max(0.0, round(ci_shelf_life[0], 2)), max(0.0, round(ci_shelf_life[1], 2))],
            "mean_degradation_score": round(min(1.0, max(0.0, mean_degradation)), 3),
            "std_degradation_score": round(std_degradation, 3),
            "mean_latent_defect_prob": round(min(1.0, max(0.0, mean_latent_defect)), 3),
            "predictive_entropy": round(predictive_entropy, 3),
            "normalized_entropy": round(norm_entropy, 3),
            "reliability": reliability,
            "reliability_badge": reliability_badge,
            "is_high_uncertainty": is_high_uncertainty,
            "mc_samples_collected": self.num_samples,
        }
