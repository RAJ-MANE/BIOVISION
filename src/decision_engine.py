"""
BioVision Retail Decision Engine.
Translates neural predictions and uncertainty intervals into actionable commercial decisions:
FEFO priority rankings, dynamic markdown triggers, store workflows, and economic impact calculations.
"""
from typing import Dict, Any, List, Optional
import os
import yaml
from pathlib import Path


class DecisionEngine:
    """
    Evaluates produce scan predictions against commercial retail policies.
    Determines FEFO prioritization, dynamic pricing interventions, and waste savings.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            # Fallback default configuration
            self.config = {
                "economics": {
                    "default_wholesale_cost_per_kg": 2.10,
                    "default_retail_price_per_kg": 4.80,
                    "default_units_per_box": 24,
                    "average_weight_kg_per_unit": 0.20,
                },
                "fefo": {
                    "weights": {
                        "shelf_life_remaining": 0.55,
                        "degradation_score": 0.30,
                        "batch_age_days": 0.15,
                    }
                },
            }

    def compute_fefo_priority(
        self,
        shelf_life_days: float,
        degradation_score: float,
        batch_age_days: float = 3.0,
        max_shelf_life: float = 14.0,
    ) -> float:
        """
        Calculate First-Expired-First-Out (FEFO) priority score on a 0 - 100 scale.
        Higher score indicates higher urgency to move or process the inventory.
        """
        weights = self.config.get("fefo", {}).get("weights", {})
        w_sl = weights.get("shelf_life_remaining", 0.55)
        w_deg = weights.get("degradation_score", 0.30)
        w_age = weights.get("batch_age_days", 0.15)

        norm_sl_urgency = 1.0 - min(1.0, max(0.0, shelf_life_days / max_shelf_life))
        norm_deg = min(1.0, max(0.0, degradation_score))
        norm_age = min(1.0, max(0.0, batch_age_days / max_shelf_life))

        score = 100.0 * (w_sl * norm_sl_urgency + w_deg * norm_deg + w_age * norm_age)
        return round(float(score), 1)

    def evaluate_action(
        self,
        shelf_life_days: float,
        degradation_score: float,
        is_high_uncertainty: bool = False,
        latent_defect_prob: float = 0.0,
        ci_95_shelf_life: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Determine the optimal commercial store action based on model predictions.
        Incorporates risk bounds and early spectral warnings.
        """
        # If prediction variance is wide, recommend human inspection first
        if is_high_uncertainty:
            return {
                "action_code": "MANUAL_INSPECTION",
                "action_name": "Physical Firmness Verification Required",
                "badge_color": "#8b5cf6",  # Purple
                "suggested_discount": 0.0,
                "urgency_level": "MODERATE",
                "reasoning": (
                    f"Model epistemic uncertainty is elevated (CI: {ci_95_shelf_life or 'wide'}). "
                    "Manual tactile inspection recommended prior to automated markdown."
                ),
                "is_automated_eligible": False,
            }

        # Severe spoilage / dead shelf-life
        if shelf_life_days <= 1.0 or degradation_score >= 0.85:
            return {
                "action_code": "DISCARD",
                "action_name": "Immediate Removal & Compost",
                "badge_color": "#ef4444",  # Red
                "suggested_discount": 1.0,
                "urgency_level": "CRITICAL",
                "reasoning": (
                    f"Critical breakdown reached (Shelf-life: {shelf_life_days:.1f}d, Degradation: {degradation_score:.2f}). "
                    "Remove from retail shelf immediately to protect customer experience."
                ),
                "is_automated_eligible": True,
            }

        # Near end of life, suitable for deli or guacamole kitchen
        if shelf_life_days <= 2.5 or degradation_score >= 0.65:
            return {
                "action_code": "REPURPOSE",
                "action_name": "Route to Deli / Kitchen Prep",
                "badge_color": "#f97316",  # Orange
                "suggested_discount": 0.60,
                "urgency_level": "HIGH",
                "reasoning": (
                    f"Fruit is fully soft and ripe (Shelf-life: {shelf_life_days:.1f}d). "
                    "Ideal for immediate kitchen processing (salads, guacamole, juice) to recover 60%+ value."
                ),
                "is_automated_eligible": True,
            }

        # Fast markdown needed (2-4 days remaining)
        if shelf_life_days <= 4.0 or degradation_score >= 0.50:
            return {
                "action_code": "MARKDOWN_50",
                "action_name": "Apply 50% Dynamic Markdown",
                "badge_color": "#eab308",  # Amber
                "suggested_discount": 0.50,
                "urgency_level": "HIGH",
                "reasoning": (
                    f"Sub-surface softening active (Shelf-life: {shelf_life_days:.1f}d). "
                    "Apply 50% promotional pricing to accelerate consumer sell-through within 48h."
                ),
                "is_automated_eligible": True,
            }

        # Early spectral alert (4-6 days, or latent internal defect detected by NIR)
        if shelf_life_days <= 6.0 or degradation_score >= 0.35 or latent_defect_prob > 0.45:
            extra_note = " Latent subsurface density drop flagged by NIR." if latent_defect_prob > 0.45 else ""
            return {
                "action_code": "MARKDOWN_20",
                "action_name": "Apply 20% Quick-Sale Markdown",
                "badge_color": "#3b82f6",  # Blue
                "suggested_discount": 0.20,
                "urgency_level": "MEDIUM",
                "reasoning": (
                    f"Early degradation phase detected (Shelf-life: {shelf_life_days:.1f}d).{extra_note} "
                    "Prioritize front-of-shelf positioning with mild 20% markdown."
                ),
                "is_automated_eligible": True,
            }

        # Prime condition
        return {
            "action_code": "MAINTAIN",
            "action_name": "Maintain Full Price / Premium Display",
            "badge_color": "#10b981",  # Green
            "suggested_discount": 0.00,
            "urgency_level": "NORMAL",
            "reasoning": (
                f"Specimen cellular density intact (Shelf-life: {shelf_life_days:.1f}d, Degradation: {degradation_score:.2f}). "
                "Retain premium retail shelf position."
            ),
            "is_automated_eligible": True,
        }

    def calculate_economics(
        self,
        batch_units: int,
        action_code: str,
        wholesale_cost_per_unit: Optional[float] = None,
        retail_price_per_unit: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate financial outcome: revenue realized, loss avoided, and profit impact.
        """
        unit_cost = wholesale_cost_per_unit or (
            self.config["economics"]["default_wholesale_cost_per_kg"] *
            self.config["economics"]["average_weight_kg_per_unit"]
        )
        unit_retail = retail_price_per_unit or (
            self.config["economics"]["default_retail_price_per_kg"] *
            self.config["economics"]["average_weight_kg_per_unit"]
        )

        batch_cost = batch_units * unit_cost
        full_revenue = batch_units * unit_retail

        # Discount schedules
        discounts = {
            "MAINTAIN": 0.0,
            "MARKDOWN_20": 0.20,
            "MARKDOWN_50": 0.50,
            "REPURPOSE": 0.40,      # In-store deli buys at 40% discount from retail
            "DISCARD": 1.00,
            "MANUAL_INSPECTION": 0.0,
        }

        discount = discounts.get(action_code, 0.0)
        actual_selling_price = unit_retail * (1.0 - discount)

        if action_code == "DISCARD":
            realized_revenue = 0.0
            net_loss = batch_cost
            loss_avoided = 0.0
        else:
            realized_revenue = batch_units * actual_selling_price
            net_loss = 0.0
            # Loss avoided vs complete spoilage dump
            loss_avoided = realized_revenue

        profit = realized_revenue - batch_cost

        return {
            "batch_units": batch_units,
            "batch_cost": round(batch_cost, 2),
            "potential_full_revenue": round(full_revenue, 2),
            "realized_revenue": round(realized_revenue, 2),
            "profit": round(profit, 2),
            "loss_avoided_vs_spoilage": round(loss_avoided, 2),
            "discount_applied_pct": round(discount * 100, 1),
        }
