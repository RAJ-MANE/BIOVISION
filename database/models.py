"""
SQLAlchemy ORM Data Models for BioVision.
Tracks inventory batches, individual spectral scans, store intervention actions, and audit logs.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Batch(Base):
    """Represents a produce inventory shipment or pallet."""
    __tablename__ = "batches"

    batch_id = Column(String(50), primary_key=True, index=True)
    produce_type = Column(String(50), default="Avocado (Hass)", nullable=False)
    variety = Column(String(50), default="Hass")
    supplier = Column(String(100), default="Mission Produce Ltd.")
    received_date = Column(DateTime, default=datetime.utcnow)
    total_units = Column(Integer, default=120)
    remaining_units = Column(Integer, default=120)
    storage_location = Column(String(50), default="Cold Room A - Zone 2")
    current_status = Column(String(30), default="ACTIVE")  # ACTIVE, HOLD, MARKDOWN, CLOSED
    
    # Aggregated freshness metrics
    fefo_priority = Column(Float, default=50.0)
    avg_shelf_life_days = Column(Float, default=7.0)
    avg_degradation_score = Column(Float, default=0.25)
    last_scanned_at = Column(DateTime, nullable=True)

    # Relationships
    scans = relationship("Scan", back_populates="batch", cascade="all, delete-orphan")
    actions = relationship("StoreAction", back_populates="batch", cascade="all, delete-orphan")


class Scan(Base):
    """Represents a single optical scan (RGB + NIR paired data) of a produce unit."""
    __tablename__ = "scans"

    scan_id = Column(String(50), primary_key=True, index=True)
    batch_id = Column(String(50), ForeignKey("batches.batch_id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    rgb_image_path = Column(String(255), nullable=True)
    nir_image_path = Column(String(255), nullable=True)

    # Neural predictions
    predicted_stage = Column(Integer, nullable=False)  # 1 to 5
    predicted_stage_name = Column(String(50), nullable=False)
    predicted_shelf_life_days = Column(Float, nullable=False)
    ci_lower_days = Column(Float, nullable=False)
    ci_upper_days = Column(Float, nullable=False)
    degradation_score = Column(Float, nullable=False)
    latent_defect_prob = Column(Float, default=0.0)
    spectral_contrast_index = Column(Float, default=0.0)

    # Uncertainty & Quality
    reliability = Column(String(30), default="HIGH_CONFIDENCE")
    predictive_entropy = Column(Float, default=0.1)

    batch = relationship("Batch", back_populates="scans")
    action = relationship("StoreAction", back_populates="scan", uselist=False)


class StoreAction(Base):
    """Store floor action recommended and tracked by the decision engine."""
    __tablename__ = "store_actions"

    action_id = Column(String(50), primary_key=True, index=True)
    batch_id = Column(String(50), ForeignKey("batches.batch_id"), nullable=False, index=True)
    scan_id = Column(String(50), ForeignKey("scans.scan_id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    action_code = Column(String(50), nullable=False)   # MAINTAIN, MARKDOWN_20, MARKDOWN_50, REPURPOSE, DISCARD, MANUAL_INSPECTION
    action_name = Column(String(100), nullable=False)
    discount_applied = Column(Float, default=0.0)
    urgency_level = Column(String(20), default="NORMAL")
    status = Column(String(30), default="PENDING")     # PENDING, APPROVED, REJECTED, EXECUTED
    
    executed_by = Column(String(100), default="AI Recommendation Engine")
    approval_notes = Column(Text, nullable=True)
    audit_reason = Column(Text, nullable=False)
    estimated_savings = Column(Float, default=0.0)

    batch = relationship("Batch", back_populates="actions")
    scan = relationship("Scan", back_populates="action")


class AuditLog(Base):
    """Immutable audit log tracking all approvals, overrides, and retail actions."""
    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    event_type = Column(String(50), nullable=False)    # SCAN_LOGGED, ACTION_PROPOSED, ACTION_APPROVED, ACTION_REJECTED, MANUAL_OVERRIDE
    entity_id = Column(String(50), nullable=False)
    operator = Column(String(100), default="Store Associate")
    details = Column(Text, nullable=False)
