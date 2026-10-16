"""
Unit tests for BioVision retail decision engine policies and economics calculations.
"""
import pytest
from src.decision_engine import DecisionEngine


def test_fefo_priority_calculation():
    engine = DecisionEngine()

    # Expiring specimen (short shelf-life, high degradation, older batch) -> high FEFO priority
    crit_score = engine.compute_fefo_priority(shelf_life_days=1.0, degradation_score=0.9, batch_age_days=6.0)
    assert crit_score >= 70.0
    assert crit_score <= 100.0

    # Fresh specimen (long shelf-life, low degradation, newly arrived) -> low FEFO priority
    fresh_score = engine.compute_fefo_priority(shelf_life_days=12.0, degradation_score=0.1, batch_age_days=1.0)
    assert fresh_score <= 35.0
    assert fresh_score >= 0.0


def test_evaluate_action_policies():
    engine = DecisionEngine()

    # High uncertainty trigger
    uncert_action = engine.evaluate_action(
        shelf_life_days=3.0,
        degradation_score=0.4,
        is_high_uncertainty=True,
    )
    assert uncert_action["action_code"] == "MANUAL_INSPECTION"
    assert not uncert_action["is_automated_eligible"]

    # Critical spoilage -> Discard
    discard_action = engine.evaluate_action(
        shelf_life_days=0.8,
        degradation_score=0.90,
    )
    assert discard_action["action_code"] == "DISCARD"
    assert discard_action["urgency_level"] == "CRITICAL"

    # Fully ripe / soft -> Repurpose to Deli / Kitchen
    repurpose_action = engine.evaluate_action(
        shelf_life_days=2.0,
        degradation_score=0.70,
    )
    assert repurpose_action["action_code"] == "REPURPOSE"
    assert repurpose_action["suggested_discount"] == 0.60

    # Fast markdown -> 50%
    md50_action = engine.evaluate_action(
        shelf_life_days=3.5,
        degradation_score=0.52,
    )
    assert md50_action["action_code"] == "MARKDOWN_50"

    # Mild early alert -> 20%
    md20_action = engine.evaluate_action(
        shelf_life_days=5.0,
        degradation_score=0.38,
    )
    assert md20_action["action_code"] == "MARKDOWN_20"

    # Prime condition -> Maintain
    maintain_action = engine.evaluate_action(
        shelf_life_days=8.5,
        degradation_score=0.15,
    )
    assert maintain_action["action_code"] == "MAINTAIN"
    assert maintain_action["suggested_discount"] == 0.0


def test_calculate_economics():
    engine = DecisionEngine()

    econ_md20 = engine.calculate_economics(batch_units=100, action_code="MARKDOWN_20")
    assert econ_md20["batch_units"] == 100
    assert econ_md20["realized_revenue"] > 0
    assert econ_md20["discount_applied_pct"] == 20.0

    econ_discard = engine.calculate_economics(batch_units=100, action_code="DISCARD")
    assert econ_discard["realized_revenue"] == 0.0
    assert econ_discard["profit"] < 0
