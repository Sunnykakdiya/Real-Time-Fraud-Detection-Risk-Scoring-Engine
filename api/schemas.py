"""
Pydantic v2 Request & Response Schemas for Fraud Risk Engine API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Payload representing an incoming payment transaction for real-time scoring."""

    TransactionAmt: float = Field(..., gt=0.0, description="Transaction amount in USD", examples=[149.99])
    ProductCD: str = Field(default="W", description="Product code: W (Retail), C (Payment), R, H, S", examples=["W"])
    card1: int = Field(default=10000, ge=1000, le=99999, description="Payment card issuer ID", examples=[10045])
    card2: Optional[float] = Field(default=321.0, description="Card bank code", examples=[321.0])
    card3: Optional[float] = Field(default=150.0, description="Card country code", examples=[150.0])
    card4: Optional[str] = Field(default="visa", description="Card brand: visa, mastercard, discover, etc.", examples=["visa"])
    card5: Optional[float] = Field(default=226.0, description="Card sub-type", examples=[226.0])
    card6: Optional[str] = Field(default="debit", description="Card type: debit, credit", examples=["debit"])
    P_emaildomain: Optional[str] = Field(default="gmail.com", description="Purchaser email domain", examples=["gmail.com"])
    R_emaildomain: Optional[str] = Field(default=None, description="Recipient email domain", examples=[None])
    DeviceType: Optional[str] = Field(default="desktop", description="Device type: desktop, mobile", examples=["desktop"])
    Transaction_Hour: Optional[int] = Field(default=14, ge=0, le=23, description="Hour of transaction in UTC (0-23)", examples=[14])
    TransactionDT: Optional[int] = Field(default=None, description="Seconds from epoch (optional)", examples=[86400])
    custom_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional custom block threshold override", examples=[0.05])

    @field_validator("TransactionAmt")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TransactionAmt must be strictly positive")
        return round(v, 2)


class RiskFactor(BaseModel):
    """Human-readable explanation of an individual feature's contribution."""

    feature: str = Field(..., description="Engineered feature identifier")
    raw_value: Union[float, str] = Field(..., description="Raw or formatted feature value")
    shap_impact: float = Field(..., description="Local SHAP value contribution")
    direction: str = Field(..., description="'increases_risk' or 'decreases_risk'")
    explanation: str = Field(..., description="Plain-English explanation")


class TransactionResponse(BaseModel):
    """Decision output including calibrated fraud probability, action tier, and SHAP drivers."""

    transaction_id: str
    fraud_probability: float
    risk_score: float
    risk_tier: str
    recommended_action: str
    threshold_used: float
    review_threshold: float
    latency_ms: float
    top_risk_factors: List[RiskFactor]
    timestamp: str


class HealthResponse(BaseModel):
    """API health status and active production model configuration."""

    status: str
    active_model: str
    is_calibrated: bool
    active_threshold: float
    review_threshold: float
    feature_count: int
    uptime_seconds: float
    server_timestamp: str


class MetricsResponse(BaseModel):
    """Offline benchmark metrics and comparison table."""

    models: Dict[str, Any]
    best_model: str
    primary_metric: str
    best_pr_auc: float
    best_roc_auc: float
    brier_score: float
    net_cost_savings: float
    savings_pct: float
    updated_at: str


class DriftResponse(BaseModel):
    """Feature-level PSI and KS drift report."""

    timestamp: str
    overall_status: str
    total_features_monitored: int
    stable_count: int
    moderate_drift_count: int
    high_drift_count: int
    features: List[Dict[str, Any]]


class AuditLogEntry(BaseModel):
    """Audit log record for a previously scored transaction."""

    transaction_id: str
    timestamp: str
    amount: float
    card1: int
    product_cd: str
    fraud_probability: float
    risk_tier: str
    action: str
    latency_ms: float
    top_factor: str
