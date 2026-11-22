"""
Explainability Module using SHAP TreeExplainer.

Wraps TreeExplainer for real-time inference, extracting the top human-readable risk drivers
with directionality and plain-English operational narratives for fraud analysts.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import shap


class FraudExplainer:
    """Wrapper around SHAP TreeExplainer for fast per-transaction local attribution."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

        # Extract base tree estimator if wrapped in CalibratedClassifierCV
        underlying_model = model
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            underlying_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            underlying_model = model.estimator

        try:
            self.explainer = shap.TreeExplainer(underlying_model)
        except Exception as e:
            # Fallback to general explainer if needed
            print(f"[!] Falling back to general Explainer: {e}")
            self.explainer = shap.Explainer(underlying_model)

    def explain_transaction(
        self,
        features_df: pd.DataFrame,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Compute top risk factors and human-readable explanations for a single transaction.
        """
        # Ensure column ordering matches feature_names
        X = features_df[self.feature_names]

        shap_values = self.explainer.shap_values(X)

        # Handle binary classification output formats from LightGBM/XGBoost/SHAP
        if isinstance(shap_values, list):
            # For multi-class or list of [class_0, class_1], take class 1 (fraud)
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = np.array(shap_values).flatten()

        raw_vals = X.iloc[0].to_dict()

        # Rank features by absolute SHAP impact
        ranked_indices = np.argsort(np.abs(sv))[::-1][:top_k]

        risk_factors: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            feat_name = self.feature_names[idx]
            impact = float(sv[idx])
            val = raw_vals.get(feat_name, 0.0)
            direction = "increases_risk" if impact > 0 else "decreases_risk"

            explanation = self._generate_narrative(feat_name, val, impact)

            risk_factors.append({
                "feature": feat_name,
                "raw_value": round(float(val), 4) if isinstance(val, (int, float, np.floating)) else str(val),
                "shap_impact": round(impact, 4),
                "direction": direction,
                "explanation": explanation,
            })

        return risk_factors

    def _generate_narrative(self, feat: str, val: Any, impact: float) -> str:
        """Translate feature name, value, and impact into clear plain-English explanation."""
        num_val = float(val) if isinstance(val, (int, float, np.floating)) else 0.0
        is_risk = impact > 0

        if feat == "amt_to_card_mean_ratio":
            if is_risk:
                return f"Spend amount is {num_val:.1f}x higher than historical average on this card."
            return f"Spend amount is consistent with normal historical card patterns ({num_val:.1f}x)."

        if feat == "amt_to_card_zscore":
            if is_risk:
                return f"Amount z-score ({num_val:+.1f}σ) indicates significant positive deviation from card baseline."
            return f"Amount is within normal statistical variance for this card ({num_val:+.1f}σ)."

        if feat in ["TransactionAmt", "log_transaction_amt"]:
            if is_risk:
                return f"Elevated purchase amount (${num_val:,.2f}) poses heightened financial loss risk."
            return f"Moderate purchase value (${num_val:,.2f}) aligns with typical retail orders."

        if feat == "time_since_last_card_txn":
            if is_risk and num_val < 300:
                return f"Rapid succession ({num_val:.0f}s since last transaction) suggests card velocity burst."
            return f"Adequate temporal gap ({num_val / 3600:.1f} hrs) since previous transaction."

        if feat == "card_txn_count":
            if is_risk and num_val <= 2:
                return f"New or infrequently used card profile (historical transactions: {num_val:.0f})."
            return f"Established account profile with {num_val:.0f} historical transactions."

        if feat == "transaction_hour":
            if is_risk:
                return f"Transaction initiated at {int(num_val):02d}:00 UTC (off-peak fraud window)."
            return f"Normal daylight operating hours ({int(num_val):02d}:00 UTC)."

        if feat == "email_domain_match":
            if is_risk and num_val == 0.0:
                return "Purchaser email domain and recipient domain do not match or are missing."
            return "Purchaser and recipient email domains are verified and identical."

        if feat == "P_emaildomain_freq":
            if is_risk:
                return "Email provider has low overall platform prevalence (rare/disposable domain)."
            return "Email domain is a recognized, reputable primary provider."

        if feat == "iforest_anomaly_score":
            if is_risk:
                return f"Unsupervised Isolation Forest detected anomalous multivariate pattern (score: {num_val:.2f})."
            return f"Normal unsupervised behavioral profile (anomaly score: {num_val:.2f})."

        if feat in ["C1", "C2", "C13"]:
            if is_risk:
                return f"High authorization counter value ({feat}={num_val:.0f}) indicates repetitive attempts."
            return f"Standard authorization counter ({feat}={num_val:.0f})."

        # General fallback
        direction_word = "elevates" if is_risk else "moderates"
        return f"{feat} (value: {num_val:.2f}) {direction_word} the model risk score."
