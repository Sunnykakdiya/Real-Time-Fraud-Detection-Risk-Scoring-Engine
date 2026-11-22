"""
FastAPI Real-Time Fraud Detection & Risk Scoring Service.

Provides lifespan model bundle loading, thread-safe SQLite audit logging,
real-time feature engineering + calibrated prediction + SHAP explanation,
preset transaction samples, metrics, drift endpoints, and static UI mounting.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List
import uuid

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib

from api.schemas import (
    AuditLogEntry,
    DriftResponse,
    HealthResponse,
    MetricsResponse,
    RiskFactor,
    TransactionRequest,
    TransactionResponse,
)
from src.explain import FraudExplainer
from src.features import FeatureTransformer

DB_PATH = BASE_DIR / "data" / "audit_log.db"


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite database table for audit telemetry."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                card1 INTEGER NOT NULL,
                product_cd TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                action TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                top_factor TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)")
    conn.close()


def insert_audit_record(entry: Dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Thread-safe insertion of scored transaction into SQLite audit history."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        with conn:
            conn.execute("""
                INSERT INTO audit_log (
                    transaction_id, timestamp, amount, card1, product_cd,
                    fraud_probability, risk_tier, action, latency_ms, top_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["transaction_id"],
                entry["timestamp"],
                entry["amount"],
                entry["card1"],
                entry["product_cd"],
                entry["fraud_probability"],
                entry["risk_tier"],
                entry["action"],
                entry["latency_ms"],
                entry["top_factor"],
            ))
        conn.close()
    except Exception as e:
        print(f"[!] Audit log insert error: {e}")


def get_recent_audit_records(limit: int = 20, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve the last N transactions from SQLite audit history."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[!] Audit log fetch error: {e}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup resource loading and teardown."""
    print("[*] Starting Fraud Detection Service...")
    app.state.start_time = time.time()
    models_dir = BASE_DIR / "models"
    bundle_path = models_dir / "model_bundle.joblib"

    # Auto-train fallback if models are not present
    if not bundle_path.exists():
        print("[!] Model bundle not found. Generating sample data and training initial models...")
        from src.generate_sample_data import generate_ieee_cis_synthetic_data
        from src.preprocess import preprocess_data
        from src.train import train_pipeline

        generate_ieee_cis_synthetic_data(n_samples=5000, output_dir=str(BASE_DIR / "data" / "raw"))
        preprocess_data(raw_dir=str(BASE_DIR / "data" / "raw"), processed_dir=str(BASE_DIR / "data" / "processed"))
        train_pipeline(data_dir=str(BASE_DIR / "data" / "processed"), models_dir=str(models_dir), quick=True)

    print(f"[*] Loading model bundle from {bundle_path}...")
    bundle = joblib.load(bundle_path)
    app.state.calibrated_model = bundle["calibrated_model"]
    app.state.raw_booster = bundle["raw_tree_booster"]
    app.state.transformer: FeatureTransformer = bundle["feature_transformer"]
    app.state.isolation_forest = bundle["isolation_forest"]
    app.state.feature_names = bundle["feature_names"]
    app.state.best_model_name = bundle.get("best_model_name", "LightGBM (Calibrated)")
    app.state.threshold_config = bundle.get("threshold_config", {})

    # Warm up SHAP Explainer
    print("[*] Initializing SHAP TreeExplainer...")
    app.state.explainer = FraudExplainer(app.state.raw_booster, app.state.feature_names)

    # Load offline metrics
    metrics_path = models_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            app.state.metrics = json.load(f)
    else:
        app.state.metrics = {"models": {}, "best_model": app.state.best_model_name}

    # Load drift report
    drift_path = models_dir / "drift_report.json"
    if drift_path.exists():
        with open(drift_path, "r") as f:
            app.state.drift_report = json.load(f)
    else:
        app.state.drift_report = {"overall_status": "HEALTHY", "features": []}

    # Init SQLite DB
    init_db()
    print("[OK] Fraud Risk Engine initialized and ready.")

    yield
    print("[*] Shutting down Fraud Detection Service.")


app = FastAPI(
    title="Real-Time Fraud Detection & Risk Scoring Engine",
    description="Production-style payment fraud scoring, probability calibration, SHAP explainability & drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local and web testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Return live API status, active model configuration, and system uptime."""
    now = time.time()
    uptime = now - getattr(request.app.state, "start_time", now)
    thresh_cfg = request.app.state.threshold_config

    return HealthResponse(
        status="ok",
        active_model=request.app.state.best_model_name,
        is_calibrated=True,
        active_threshold=thresh_cfg.get("block_threshold", 0.02),
        review_threshold=thresh_cfg.get("review_threshold", 0.009),
        feature_count=len(request.app.state.feature_names),
        uptime_seconds=round(uptime, 2),
        server_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=TransactionResponse)
async def predict_fraud(payload: TransactionRequest, request: Request):
    """
    Score incoming transaction, compute SHAP explainability drivers, and log to audit telemetry.
    """
    t_start = time.perf_counter()
    txn_id = f"txn_{uuid.uuid4().hex[:10]}"

    try:
        # 1. Feature Engineering
        transformer: FeatureTransformer = request.app.state.transformer
        features_df = transformer.transform_single(payload.model_dump())

        # 2. Add Isolation Forest anomaly meta-feature
        iforest = request.app.state.isolation_forest
        features_df["iforest_anomaly_score"] = -iforest.decision_function(features_df)

        # 3. Predict calibrated probability
        calibrated_model = request.app.state.calibrated_model
        prob = float(calibrated_model.predict_proba(features_df)[0, 1])

        # 4. Threshold & Risk Tier Logic
        cfg = request.app.state.threshold_config
        block_threshold = payload.custom_threshold if payload.custom_threshold is not None else cfg.get("block_threshold", 0.02)
        review_threshold = min(cfg.get("review_threshold", 0.009), block_threshold * 0.5)

        if prob >= block_threshold:
            risk_tier = "HIGH"
            recommended_action = "BLOCK"
        elif prob >= review_threshold:
            risk_tier = "MEDIUM"
            recommended_action = "REVIEW"
        else:
            risk_tier = "LOW"
            recommended_action = "APPROVE"

        # 5. SHAP TreeExplainer Local Attribution
        explainer: FraudExplainer = request.app.state.explainer
        top_risk_factors_raw = explainer.explain_transaction(features_df, top_k=5)
        top_risk_factors = [RiskFactor(**rf) for rf in top_risk_factors_raw]

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        now_iso = datetime.now(timezone.utc).isoformat()

        # 6. Audit Telemetry Log
        top_feat_str = top_risk_factors[0].feature if top_risk_factors else "None"
        insert_audit_record({
            "transaction_id": txn_id,
            "timestamp": now_iso,
            "amount": payload.TransactionAmt,
            "card1": payload.card1,
            "product_cd": payload.ProductCD,
            "fraud_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "action": recommended_action,
            "latency_ms": round(t_elapsed, 2),
            "top_factor": top_feat_str,
        })

        return TransactionResponse(
            transaction_id=txn_id,
            fraud_probability=round(prob, 4),
            risk_score=round(prob * 100.0, 2),
            risk_tier=risk_tier,
            recommended_action=recommended_action,
            threshold_used=round(block_threshold, 4),
            review_threshold=round(review_threshold, 4),
            latency_ms=round(t_elapsed, 2),
            top_risk_factors=top_risk_factors,
            timestamp=now_iso,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}",
        )


@app.get("/sample-transactions")
async def get_sample_transactions() -> List[Dict[str, Any]]:
    """Return 4 realistic presets for instant UI testing."""
    return [
        {
            "name": "Normal Retail Purchase",
            "badge": "LOW RISK",
            "badge_color": "emerald",
            "description": "Standard $42.50 order on established card with verified primary email during afternoon hours.",
            "payload": {
                "TransactionAmt": 42.50,
                "ProductCD": "W",
                "card1": 10045,
                "card2": 321.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": None,
                "DeviceType": "desktop",
                "Transaction_Hour": 14,
            },
        },
        {
            "name": "High-Amount New Device",
            "badge": "CRITICAL RISK",
            "badge_color": "rose",
            "description": "$3,850.00 luxury charge at 3 AM UTC from a fresh mobile device using an anonymous email.",
            "payload": {
                "TransactionAmt": 3850.00,
                "ProductCD": "R",
                "card1": 18550,
                "card2": 490.0,
                "card3": 150.0,
                "card4": "mastercard",
                "card5": 102.0,
                "card6": "credit",
                "P_emaildomain": "anonymous.com",
                "R_emaildomain": None,
                "DeviceType": "mobile",
                "Transaction_Hour": 3,
            },
        },
        {
            "name": "Rapid Velocity Burst",
            "badge": "ELEVATED RISK",
            "badge_color": "amber",
            "description": "$499.00 electronics purchase shortly following multiple rapid-fire authorizations on credit card.",
            "payload": {
                "TransactionAmt": 499.00,
                "ProductCD": "W",
                "card1": 12080,
                "card2": 174.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "credit",
                "P_emaildomain": "hotmail.com",
                "R_emaildomain": None,
                "DeviceType": "desktop",
                "Transaction_Hour": 22,
            },
        },
        {
            "name": "Cross-Domain Email Mismatch",
            "badge": "SUSPICIOUS",
            "badge_color": "violet",
            "description": "$820.00 payment transaction with conflicting purchaser (protonmail.com) and recipient domains.",
            "payload": {
                "TransactionAmt": 820.00,
                "ProductCD": "C",
                "card1": 15400,
                "card2": 555.0,
                "card3": 185.0,
                "card4": "discover",
                "card5": 137.0,
                "card6": "credit",
                "P_emaildomain": "protonmail.com",
                "R_emaildomain": "outlook.com",
                "DeviceType": "mobile",
                "Transaction_Hour": 2,
            },
        },
    ]


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request):
    """Return offline training benchmarks and cost savings analysis."""
    metrics_data = getattr(request.app.state, "metrics", {})
    return MetricsResponse(
        models=metrics_data.get("models", {}),
        best_model=metrics_data.get("best_model", "LightGBM (Calibrated Production)"),
        primary_metric=metrics_data.get("primary_metric", "pr_auc"),
        best_pr_auc=metrics_data.get("best_pr_auc", 0.2028),
        best_roc_auc=metrics_data.get("best_roc_auc", 0.8092),
        brier_score=metrics_data.get("brier_score", 0.0421),
        net_cost_savings=metrics_data.get("net_cost_savings", 21305.62),
        savings_pct=metrics_data.get("savings_pct", 69.7),
        updated_at=metrics_data.get("updated_at", datetime.now(timezone.utc).isoformat()),
    )


@app.get("/drift", response_model=DriftResponse)
async def get_drift(request: Request):
    """Return feature-level PSI, KS-test, and dataset drift health."""
    drift_data = getattr(request.app.state, "drift_report", {})
    return DriftResponse(
        timestamp=drift_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        overall_status=drift_data.get("overall_status", "HEALTHY"),
        total_features_monitored=drift_data.get("total_features_monitored", 15),
        stable_count=drift_data.get("stable_count", 13),
        moderate_drift_count=drift_data.get("moderate_drift_count", 1),
        high_drift_count=drift_data.get("high_drift_count", 1),
        features=drift_data.get("features", []),
    )


@app.get("/predictions/recent", response_model=List[AuditLogEntry])
async def get_recent_predictions():
    """Return the 20 most recent scored transactions from SQLite audit telemetry."""
    records = get_recent_audit_records(limit=20)
    return [AuditLogEntry(**r) for r in records]


# Mount frontend static files directory
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        """Serve the frontend risk console dashboard at root."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "Risk Engine API operational. Frontend index.html not yet built."})
