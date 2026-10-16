"""
Integration tests for BioVision FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from database.repository import get_db_repository


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database is seeded prior to test runs."""
    repo = get_db_repository()
    repo.seed_initial_demo_data()


def test_root_and_health():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ONLINE"

        res_h = client.get("/health")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "healthy"


def test_inventory_endpoints():
    with TestClient(app) as client:
        res = client.get("/api/v1/inventory/batches")
        assert res.status_code == 200
        batches = res.json()
        assert isinstance(batches, list)
        assert len(batches) > 0

        # Summary
        res_s = client.get("/api/v1/inventory/summary")
        assert res_s.status_code == 200
        summary = res_s.json()
        assert "total_batches" in summary
        assert "active_inventory_units" in summary


def test_scan_endpoint_synthetic():
    with TestClient(app) as client:
        res = client.post("/api/v1/scan", data={"batch_id": "BATCH-AVO-2026-081", "sample_stage": 3, "batch_units": 80})
        assert res.status_code == 200
        data = res.json()
        assert "scan_id" in data
        assert "prediction" in data
        pred = data["prediction"]
        assert "shelf_life_days" in pred
        assert "ci_95_shelf_life" in pred
        assert "action" in pred
        assert "fefo_priority" in pred


def test_actions_endpoints():
    with TestClient(app) as client:
        res = client.get("/api/v1/actions")
        assert res.status_code == 200
        actions = res.json()
        assert isinstance(actions, list)

        if actions:
            action_id = actions[0]["action_id"]
            # Approve
            res_app = client.post(f"/api/v1/actions/{action_id}/approve", json={"operator": "Test Suite", "notes": "Approved in unit test"})
            assert res_app.status_code == 200
            assert res_app.json()["status"] == "APPROVED"


def test_evidence_endpoints():
    with TestClient(app) as client:
        res = client.get("/api/v1/evidence/ablation")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        assert "summary" in data

        res_curve = client.get("/api/v1/evidence/lead-time-curve")
        assert res_curve.status_code == 200
        curve = res_curve.json()
        assert "days" in curve
        assert "early_warning_delta_days" in curve
