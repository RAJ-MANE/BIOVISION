"""
Database repository providing CRUD operations, metrics aggregation,
and realistic seed data generation for BioVision.
"""
from typing import Dict, Any, List, Optional
import os
from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Batch, Scan, StoreAction, AuditLog
from src.decision_engine import DecisionEngine


DEFAULT_DB_PATH = "biovision.db"
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH}"


class DatabaseRepository:
    """Manages transactional access to the BioVision SQLite database."""

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_db()

    def init_db(self):
        """Create all tables if they do not already exist."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # --- Batch Operations ---

    def create_batch(self, batch_data: Dict[str, Any]) -> Batch:
        with self.get_session() as session:
            batch = Batch(**batch_data)
            session.add(batch)
            session.commit()
            session.refresh(batch)
            return batch

    def get_batches(self, status: Optional[str] = None, sort_by_fefo: bool = True) -> List[Batch]:
        with self.get_session() as session:
            query = session.query(Batch)
            if status:
                query = query.filter(Batch.current_status == status)
            if sort_by_fefo:
                query = query.order_by(desc(Batch.fefo_priority))
            else:
                query = query.order_by(Batch.received_date)
            return query.all()

    def get_batch(self, batch_id: str) -> Optional[Batch]:
        with self.get_session() as session:
            return session.query(Batch).filter(Batch.batch_id == batch_id).first()

    def update_batch_freshness(
        self,
        batch_id: str,
        avg_shelf_life: float,
        avg_degradation: float,
        fefo_priority: float,
    ):
        with self.get_session() as session:
            batch = session.query(Batch).filter(Batch.batch_id == batch_id).first()
            if batch:
                batch.avg_shelf_life_days = avg_shelf_life
                batch.avg_degradation_score = avg_degradation
                batch.fefo_priority = fefo_priority
                batch.last_scanned_at = datetime.utcnow()
                session.commit()

    # --- Scan Operations ---

    def create_scan(self, scan_data: Dict[str, Any]) -> Scan:
        with self.get_session() as session:
            scan = Scan(**scan_data)
            session.add(scan)
            session.commit()
            session.refresh(scan)
            return scan

    def get_scans_by_batch(self, batch_id: str, limit: int = 50) -> List[Scan]:
        with self.get_session() as session:
            return (
                session.query(Scan)
                .filter(Scan.batch_id == batch_id)
                .order_by(desc(Scan.timestamp))
                .limit(limit)
                .all()
            )

    def get_recent_scans(self, limit: int = 25) -> List[Scan]:
        with self.get_session() as session:
            return session.query(Scan).order_by(desc(Scan.timestamp)).limit(limit).all()

    # --- Action Operations ---

    def create_action(self, action_data: Dict[str, Any]) -> StoreAction:
        with self.get_session() as session:
            action = StoreAction(**action_data)
            session.add(action)
            session.commit()
            session.refresh(action)
            return action

    def get_actions(self, status: Optional[str] = None) -> List[StoreAction]:
        with self.get_session() as session:
            query = session.query(StoreAction)
            if status:
                query = query.filter(StoreAction.status == status)
            return query.order_by(desc(StoreAction.timestamp)).all()

    def update_action_status(
        self,
        action_id: str,
        status: str,
        operator: str = "Store Manager",
        notes: Optional[str] = None,
    ) -> Optional[StoreAction]:
        with self.get_session() as session:
            action = session.query(StoreAction).filter(StoreAction.action_id == action_id).first()
            if action:
                action.status = status
                action.approval_notes = notes
                action.executed_by = operator
                
                # Log audit event
                log = AuditLog(
                    event_type=f"ACTION_{status.upper()}",
                    entity_id=action_id,
                    operator=operator,
                    details=f"Action '{action.action_name}' marked as {status}. Reason: {notes or 'Standard protocol'}",
                )
                session.add(log)
                session.commit()
                session.refresh(action)
            return action

    # --- Audit Logs ---

    def get_audit_logs(self, limit: int = 50) -> List[AuditLog]:
        with self.get_session() as session:
            return session.query(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit).all()

    # --- Summary Metrics ---

    def get_summary_metrics(self) -> Dict[str, Any]:
        with self.get_session() as session:
            total_batches = session.query(func.count(Batch.batch_id)).scalar() or 0
            active_units = session.query(func.sum(Batch.remaining_units)).filter(Batch.current_status == "ACTIVE").scalar() or 0
            pending_actions = session.query(func.count(StoreAction.action_id)).filter(StoreAction.status == "PENDING").scalar() or 0
            
            # Total loss avoided from approved actions
            total_savings = session.query(func.sum(StoreAction.estimated_savings)).filter(
                StoreAction.status.in_(["APPROVED", "EXECUTED"])
            ).scalar() or 0.0

            # High urgency batches
            critical_batches = session.query(func.count(Batch.batch_id)).filter(
                Batch.fefo_priority >= 70.0, Batch.current_status == "ACTIVE"
            ).scalar() or 0

            return {
                "total_batches": total_batches,
                "active_inventory_units": int(active_units),
                "pending_action_count": pending_actions,
                "loss_avoided_dollars": round(float(total_savings), 2),
                "critical_fefo_batches": critical_batches,
            }

    # --- Realistic Seeding ---

    def seed_initial_demo_data(self, force: bool = False):
        """Seed realistic inventory and initial actions if database is empty or force=True."""
        with self.get_session() as session:
            if force:
                session.query(AuditLog).delete()
                session.query(StoreAction).delete()
                session.query(Scan).delete()
                session.query(Batch).delete()
                session.commit()
            elif session.query(Batch).count() > 0:
                return  # Already populated

        engine_rules = DecisionEngine()
        now = datetime.utcnow()

        seed_batches = [
            {
                "batch_id": "BATCH-AVO-2026-081",
                "produce_type": "Avocado (Hass)",
                "variety": "Hass Class 1",
                "supplier": "Michoacán Premium Groves, MX",
                "received_date": now - timedelta(days=6),
                "total_units": 150,
                "remaining_units": 84,
                "storage_location": "Front Aisle - Endcap 1",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 1.8,
                "avg_degradation_score": 0.72,
                "fefo_priority": 88.5,
                "last_scanned_at": now - timedelta(hours=4),
            },
            {
                "batch_id": "BATCH-MANGO-2026-104",
                "produce_type": "Mango (Alphonso)",
                "variety": "Ratnagiri GI Alphonso",
                "supplier": "Devgad Agro Exporters, India",
                "received_date": now - timedelta(days=4),
                "total_units": 180,
                "remaining_units": 135,
                "storage_location": "Aisle 2 - Exotic Tropicals",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 2.2,
                "avg_degradation_score": 0.64,
                "fefo_priority": 82.0,
                "last_scanned_at": now - timedelta(hours=6),
            },
            {
                "batch_id": "BATCH-APPL-2026-052",
                "produce_type": "Apple (Honeycrisp)",
                "variety": "Honeycrisp Grade A",
                "supplier": "Yakima Valley Orchards, WA",
                "received_date": now - timedelta(days=8),
                "total_units": 320,
                "remaining_units": 240,
                "storage_location": "Aisle 1 - Orchard Display",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 3.9,
                "avg_degradation_score": 0.48,
                "fefo_priority": 68.5,
                "last_scanned_at": now - timedelta(hours=10),
            },
            {
                "batch_id": "BATCH-BERRY-2026-219",
                "produce_type": "Strawberry (Sweet Charlie)",
                "variety": "Sweet Charlie Primocane",
                "supplier": "Driscoll's Coastal Valley, CA",
                "received_date": now - timedelta(days=2),
                "total_units": 200,
                "remaining_units": 160,
                "storage_location": "Chilled Display Chute 3",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 1.5,
                "avg_degradation_score": 0.76,
                "fefo_priority": 91.0,
                "last_scanned_at": now - timedelta(hours=2),
            },
            {
                "batch_id": "BATCH-AVO-2026-092",
                "produce_type": "Avocado (Hass)",
                "variety": "Hass Jumbo Organic",
                "supplier": "Westfalia Fruit, Colombia",
                "received_date": now - timedelta(days=2),
                "total_units": 220,
                "remaining_units": 220,
                "storage_location": "Cold Room A - Rack 04",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 7.8,
                "avg_degradation_score": 0.22,
                "fefo_priority": 38.6,
                "last_scanned_at": now - timedelta(days=1),
            },
            {
                "batch_id": "BATCH-MANGO-2026-112",
                "produce_type": "Mango (Alphonso)",
                "variety": "Kesar Export Grade",
                "supplier": "Gujarat Agro Cooperative, India",
                "received_date": now - timedelta(hours=20),
                "total_units": 250,
                "remaining_units": 250,
                "storage_location": "Cold Vault B - Pallet 09",
                "current_status": "ACTIVE",
                "avg_shelf_life_days": 9.5,
                "avg_degradation_score": 0.15,
                "fefo_priority": 22.4,
                "last_scanned_at": now - timedelta(hours=8),
            },
        ]

        with self.get_session() as session:
            for b_info in seed_batches:
                b = Batch(**b_info)
                session.add(b)
            session.commit()

        # Seed initial pending / approved actions
        actions_to_seed = [
            {
                "action_id": "ACT-2026-001",
                "batch_id": "BATCH-AVO-2026-081",
                "timestamp": now - timedelta(hours=3),
                "action_code": "REPURPOSE",
                "action_name": "Route to Deli / Kitchen Prep",
                "discount_applied": 0.60,
                "urgency_level": "HIGH",
                "status": "PENDING",
                "executed_by": "BioVision Automated Policy",
                "audit_reason": "Shelf-life 1.8 days remaining. Route 40 units to in-store guacamole prep.",
                "estimated_savings": 64.80,
            },
            {
                "action_id": "ACT-2026-002",
                "batch_id": "BATCH-AVO-2026-085",
                "timestamp": now - timedelta(hours=6),
                "action_code": "MARKDOWN_50",
                "action_name": "Apply 50% Dynamic Markdown",
                "discount_applied": 0.50,
                "urgency_level": "HIGH",
                "status": "APPROVED",
                "executed_by": "J. Miller (Produce Lead)",
                "audit_reason": "Pre-symptomatic internal tissue density decline detected by NIR.",
                "estimated_savings": 82.50,
            },
            {
                "action_id": "ACT-2026-003",
                "batch_id": "BATCH-AVO-2026-089",
                "timestamp": now - timedelta(hours=12),
                "action_code": "MARKDOWN_20",
                "action_name": "Apply 20% Quick-Sale Markdown",
                "discount_applied": 0.20,
                "urgency_level": "MEDIUM",
                "status": "EXECUTED",
                "executed_by": "A. Vance (Store Ops)",
                "audit_reason": "Proactive 20% promotional velocity boost before day 5 soft transition.",
                "estimated_savings": 46.20,
            },
        ]

        with self.get_session() as session:
            for act_info in actions_to_seed:
                act = StoreAction(**act_info)
                session.add(act)
                
                # Add corresponding audit log
                audit = AuditLog(
                    event_type=f"ACTION_{act.status}",
                    entity_id=act.action_id,
                    operator=act.executed_by,
                    details=f"Initialized {act.action_name} for batch {act.batch_id}. Status: {act.status}",
                )
                session.add(audit)
            session.commit()


_db_repo_instance: Optional[DatabaseRepository] = None

def get_db_repository() -> DatabaseRepository:
    """Singleton getter for DatabaseRepository."""
    global _db_repo_instance
    if _db_repo_instance is None:
        _db_repo_instance = DatabaseRepository()
    return _db_repo_instance
