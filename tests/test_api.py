"""
Integration Tests for FastAPI Backend Endpoints using TestClient.
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context execution."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test GET /health returns operational status and model metadata."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "active_model" in data
    assert data["is_calibrated"] is True
    assert data["feature_count"] > 20
    assert data["active_threshold"] > 0
    assert data["uptime_seconds"] >= 0


def test_predict_normal_transaction(client):
    """Test POST /predict with a realistic legitimate retail transaction."""
    payload = {
        "TransactionAmt": 45.00,
        "ProductCD": "W",
        "card1": 10045,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "DeviceType": "desktop",
        "Transaction_Hour": 14,
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "transaction_id" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["recommended_action"] in ["APPROVE", "REVIEW", "BLOCK"]
    assert data["latency_ms"] > 0

    # Verify SHAP explainability drivers are present
    assert len(data["top_risk_factors"]) == 5
    for rf in data["top_risk_factors"]:
        assert "feature" in rf
        assert "shap_impact" in rf
        assert rf["direction"] in ["increases_risk", "decreases_risk"]
        assert len(rf["explanation"]) > 5


def test_predict_custom_threshold_override(client):
    """Test POST /predict respects a custom threshold override."""
    payload = {
        "TransactionAmt": 100.00,
        "ProductCD": "W",
        "card1": 10045,
        "card4": "visa",
        "card6": "debit",
        "custom_threshold": 0.99,  # Very high threshold, should approve
    }

    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["threshold_used"] == 0.99


def test_predict_validation_error(client):
    """Test POST /predict returns 422 for non-positive amount."""
    invalid_payload = {
        "TransactionAmt": -10.0,  # Invalid negative amount
        "ProductCD": "W",
        "card1": 10000,
    }

    res = client.post("/predict", json=invalid_payload)
    assert res.status_code == 422


def test_sample_transactions_endpoint(client):
    """Test GET /sample-transactions returns 4 realistic presets."""
    res = client.get("/sample-transactions")
    assert res.status_code == 200
    presets = res.json()

    assert len(presets) == 4
    for p in presets:
        assert "name" in p
        assert "payload" in p
        assert "TransactionAmt" in p["payload"]
        assert "badge" in p


def test_metrics_endpoint(client):
    """Test GET /metrics returns model comparison benchmark table."""
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()

    assert "models" in data
    assert len(data["models"]) >= 3
    assert data["best_pr_auc"] > 0
    assert data["best_roc_auc"] > 0.5


def test_drift_endpoint(client):
    """Test GET /drift returns feature stability indicators."""
    res = client.get("/drift")
    assert res.status_code == 200
    data = res.json()

    assert data["overall_status"] in ["HEALTHY", "WARNING", "CRITICAL"]
    assert data["total_features_monitored"] > 0
    assert isinstance(data["features"], list)


def test_recent_predictions_endpoint(client):
    """Test GET /predictions/recent retrieves logged audit entries."""
    # Score one transaction first to ensure audit log has at least one entry
    client.post("/predict", json={"TransactionAmt": 75.0, "card1": 10045})

    res = client.get("/predictions/recent")
    assert res.status_code == 200
    records = res.json()

    assert isinstance(records, list)
    assert len(records) > 0
    first = records[0]
    assert "transaction_id" in first
    assert "fraud_probability" in first
    assert "action" in first
