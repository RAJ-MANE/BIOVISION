"""
Produce optical scan endpoint.
Receives RGB and NIR image payloads, processes neural multimodal inference,
logs database records, and returns actionable retail decisions.
"""
from typing import Optional
from datetime import datetime
import uuid
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io

from api.schemas import ScanResponse, ScanPredictionResult
from src.inference import ProduceInferencePipeline
from database.repository import get_db_repository
from database.models import Scan, StoreAction, AuditLog

router = APIRouter(prefix="/scan", tags=["Scanning & Prediction"])

# Lazy singleton inference pipeline
_inference_pipeline: Optional[ProduceInferencePipeline] = None

def get_pipeline() -> ProduceInferencePipeline:
    global _inference_pipeline
    if _inference_pipeline is None:
        _inference_pipeline = ProduceInferencePipeline()
    return _inference_pipeline


@router.post("", response_model=ScanResponse)
async def process_produce_scan(
    rgb_file: Optional[UploadFile] = File(None),
    nir_file: Optional[UploadFile] = File(None),
    batch_id: str = Form("BATCH-AVO-2026-081"),
    batch_units: int = Form(100),
    sample_stage: Optional[int] = Form(None),  # Optional mock generator trigger (1-5)
):
    """
    Process paired RGB + NIR produce images.
    If files are omitted, optionally creates a synthetic sample based on sample_stage (1-5)
    for interactive API testing or automated CI pipeline runs.
    """
    pipeline = get_pipeline()
    repo = get_db_repository()

    # Load or generate images
    if rgb_file and nir_file:
        try:
            rgb_bytes = await rgb_file.read()
            nir_bytes = await nir_file.read()
            rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            nir_img = Image.open(io.BytesIO(nir_bytes)).convert("L")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image files: {str(e)}")
    else:
        # Generate synthetic specimen for demo/testing
        from src.data.loaders import SpectralProduceDataset
        target_stage = sample_stage if sample_stage is not None else 3
        degradation = 0.15 * (target_stage - 1)
        rgb_img = SpectralProduceDataset._create_synthetic_avocado_rgb(target_stage)
        nir_img = SpectralProduceDataset._create_synthetic_avocado_nir(target_stage, degradation)

    # Execute inference
    try:
        prediction = pipeline.predict(
            rgb_input=rgb_img,
            nir_input=nir_img,
            batch_id=batch_id,
            batch_units=batch_units,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

    scan_id = f"SCN-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow()

    # Save Scan to Database
    repo.create_scan({
        "scan_id": scan_id,
        "batch_id": batch_id,
        "timestamp": now,
        "predicted_stage": prediction["predicted_stage"],
        "predicted_stage_name": prediction["predicted_stage_name"],
        "predicted_shelf_life_days": prediction["shelf_life_days"],
        "ci_lower_days": prediction["ci_95_shelf_life"][0],
        "ci_upper_days": prediction["ci_95_shelf_life"][1],
        "degradation_score": prediction["degradation_score"],
        "latent_defect_prob": prediction["latent_defect_prob"],
        "spectral_contrast_index": prediction["spectral_contrast_index"],
        "reliability": prediction["reliability"],
        "predictive_entropy": prediction["predictive_entropy"],
    })

    # Save Recommended Store Action
    act_data = prediction["action"]
    action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
    repo.create_action({
        "action_id": action_id,
        "batch_id": batch_id,
        "scan_id": scan_id,
        "timestamp": now,
        "action_code": act_data["action_code"],
        "action_name": act_data["action_name"],
        "discount_applied": act_data["suggested_discount"],
        "urgency_level": act_data["urgency_level"],
        "status": "PENDING" if act_data["is_automated_eligible"] else "HOLD",
        "executed_by": "BioVision Automated Policy",
        "audit_reason": act_data["reasoning"],
        "estimated_savings": prediction["economics"]["loss_avoided_vs_spoilage"],
    })

    # Update Batch Aggregates in DB
    repo.update_batch_freshness(
        batch_id=batch_id,
        avg_shelf_life=prediction["shelf_life_days"],
        avg_degradation=prediction["degradation_score"],
        fefo_priority=prediction["fefo_priority"],
    )

    return ScanResponse(
        scan_id=scan_id,
        timestamp=now,
        prediction=ScanPredictionResult(**prediction),
        message="Scan successfully analyzed, logged, and commercial policy evaluated.",
    )
