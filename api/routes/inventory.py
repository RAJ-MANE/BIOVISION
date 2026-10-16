"""
Inventory intelligence API endpoints.
Provides FEFO-ranked batch listings, lot tracking, and warehouse summary statistics.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from api.schemas import BatchItem, SummaryMetrics
from database.repository import get_db_repository

router = APIRouter(prefix="/inventory", tags=["Inventory Intelligence"])


@router.get("/batches", response_model=List[BatchItem])
def list_inventory_batches(
    status: Optional[str] = Query(None, description="Filter by status: ACTIVE, HOLD, MARKDOWN, CLOSED"),
    sort_by_fefo: bool = Query(True, description="Sort descending by FEFO urgency score"),
):
    """Retrieve all store produce batches, ordered by FEFO priority by default."""
    repo = get_db_repository()
    batches = repo.get_batches(status=status, sort_by_fefo=sort_by_fefo)
    return batches


@router.get("/batches/{batch_id}", response_model=BatchItem)
def get_batch_details(batch_id: str):
    """Retrieve metadata and aggregated shelf-life condition for a specific lot."""
    repo = get_db_repository()
    batch = repo.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return batch


@router.get("/summary", response_model=SummaryMetrics)
def get_inventory_summary_metrics():
    """Retrieve macro KPIs: total lots, active units, pending interventions, and dollars saved."""
    repo = get_db_repository()
    metrics = repo.get_summary_metrics()
    return SummaryMetrics(**metrics)
