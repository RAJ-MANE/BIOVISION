"""
Store Action & Retail Policy Decision Endpoints.
Enables floor managers to review, approve, reject, or override AI recommendations,
with all transactions recorded in an immutable audit log.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from api.schemas import ActionItem, ActionDecisionRequest
from database.repository import get_db_repository

router = APIRouter(prefix="/actions", tags=["Action Centre & Policy Execution"])


@router.get("", response_model=List[ActionItem])
def list_store_actions(
    status: Optional[str] = Query(None, description="Filter by status: PENDING, APPROVED, REJECTED, EXECUTED")
):
    """List all AI-recommended commercial interventions."""
    repo = get_db_repository()
    return repo.get_actions(status=status)


@router.post("/{action_id}/approve", response_model=ActionItem)
def approve_store_action(action_id: str, request: ActionDecisionRequest):
    """Approve a pending recommendation (e.g. 20% markdown, kitchen repurposing)."""
    repo = get_db_repository()
    updated = repo.update_action_status(
        action_id=action_id,
        status="APPROVED",
        operator=request.operator,
        notes=request.notes or "Approved per standard retail FEFO guidelines.",
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
    return updated


@router.post("/{action_id}/reject", response_model=ActionItem)
def reject_store_action(action_id: str, request: ActionDecisionRequest):
    """Reject or override an AI recommendation with required manager notes."""
    repo = get_db_repository()
    if not request.notes:
        raise HTTPException(status_code=400, detail="Manager override reasoning note is required to reject an action.")
    
    updated = repo.update_action_status(
        action_id=action_id,
        status="REJECTED",
        operator=request.operator,
        notes=request.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
    return updated


@router.get("/audit-log")
def get_compliance_audit_log(limit: int = Query(50, ge=1, le=200)):
    """Retrieve immutable chronological audit records of all produce decisions."""
    repo = get_db_repository()
    logs = repo.get_audit_logs(limit=limit)
    return [
        {
            "log_id": log.log_id,
            "timestamp": log.timestamp.isoformat(),
            "event_type": log.event_type,
            "entity_id": log.entity_id,
            "operator": log.operator,
            "details": log.details,
        }
        for log in logs
    ]
