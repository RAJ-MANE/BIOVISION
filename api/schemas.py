"""
Pydantic Schemas for BioVision API endpoints.
Defines strict input validation and serialized response models.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ScanPredictionResult(BaseModel):
    batch_id: str
    predicted_stage: int
    predicted_stage_name: str
    class_probabilities: List[float]
    shelf_life_days: float
    shelf_life_std: float
    ci_95_shelf_life: List[float]
    degradation_score: float
    latent_defect_prob: float
    spectral_contrast_index: float
    reliability: str
    reliability_badge: str
    predictive_entropy: float
    normalized_entropy: float
    is_high_uncertainty: bool
    fefo_priority: float
    action: Dict[str, Any]
    economics: Dict[str, Any]


class ScanResponse(BaseModel):
    scan_id: str
    timestamp: datetime
    prediction: ScanPredictionResult
    message: str


class BatchItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    batch_id: str
    produce_type: str
    variety: str
    supplier: str
    received_date: datetime
    total_units: int
    remaining_units: int
    storage_location: str
    current_status: str
    fefo_priority: float
    avg_shelf_life_days: float
    avg_degradation_score: float
    last_scanned_at: Optional[datetime] = None


class ActionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    action_id: str
    batch_id: str
    scan_id: Optional[str] = None
    timestamp: datetime
    action_code: str
    action_name: str
    discount_applied: float
    urgency_level: str
    status: str
    executed_by: str
    approval_notes: Optional[str] = None
    audit_reason: str
    estimated_savings: float


class ActionDecisionRequest(BaseModel):
    operator: str = Field(default="Store Manager", description="Name/role of staff member executing decision")
    notes: Optional[str] = Field(default=None, description="Optional override reason or notes")


class SummaryMetrics(BaseModel):
    total_batches: int
    active_inventory_units: int
    pending_action_count: int
    loss_avoided_dollars: float
    critical_fefo_batches: int


class AblationResponse(BaseModel):
    models: Dict[str, Any]
    summary: Dict[str, Any]
