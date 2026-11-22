"""
Data & Feature Drift Monitoring Module.

Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS)
tests comparing production/holdout distributions against training baselines to detect
covariate shift and model degradation.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def compute_psi(
    reference: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between reference and target arrays.
      PSI < 0.10: No significant change (STABLE)
      0.10 <= PSI < 0.25: Moderate shift (MODERATE_DRIFT)
      PSI >= 0.25: Significant distributional shift (HIGH_DRIFT)
    """
    ref_clean = reference[~np.isnan(reference)]
    target_clean = target[~np.isnan(target)]

    if len(ref_clean) < 10 or len(target_clean) < 10:
        return 0.0

    # Quantile bins from reference
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Ensure min and max bounds encapsulate all target data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    target_counts, _ = np.histogram(target_clean, bins=bin_edges)

    # Convert to proportions with smoothing
    ref_prop = ref_counts / max(len(ref_clean), 1)
    target_prop = target_counts / max(len(target_clean), 1)

    ref_prop = np.where(ref_prop == 0, epsilon, ref_prop)
    target_prop = np.where(target_prop == 0, epsilon, target_prop)

    # PSI equation
    psi_val = np.sum((target_prop - ref_prop) * np.log(target_prop / ref_prop))
    return float(np.round(psi_val, 4))


def compute_ks_test(
    reference: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """Calculate Two-Sample Kolmogorov-Smirnov test statistic and p-value."""
    ref_clean = reference[~np.isnan(reference)]
    target_clean = target[~np.isnan(target)]

    if len(ref_clean) < 5 or len(target_clean) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0}

    ks_stat, p_val = stats.ks_2samp(ref_clean, target_clean)
    return {"ks_statistic": round(float(ks_stat), 4), "p_value": round(float(p_val), 4)}


def analyze_drift(
    ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: List[str] | None = None,
    output_path: str = "models/drift_report.json",
) -> Dict[str, Any]:
    """
    Compute feature-level PSI, KS-test, and overall health assessment.
    """
    if features is None:
        features = [
            c for c in ref_df.columns
            if c in target_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])
        ]

    feature_reports: List[Dict[str, Any]] = []
    high_drift_count = 0
    moderate_drift_count = 0

    for feat in features:
        ref_vals = ref_df[feat].values.astype(float)
        tar_vals = target_df[feat].values.astype(float)

        psi = compute_psi(ref_vals, tar_vals)
        ks_res = compute_ks_test(ref_vals, tar_vals)

        if psi >= 0.25:
            status = "HIGH_DRIFT"
            high_drift_count += 1
        elif psi >= 0.10:
            status = "MODERATE_DRIFT"
            moderate_drift_count += 1
        else:
            status = "STABLE"

        feature_reports.append({
            "feature": feat,
            "psi": psi,
            "ks_statistic": ks_res["ks_statistic"],
            "p_value": ks_res["p_value"],
            "status": status,
            "ref_mean": round(float(np.nanmean(ref_vals)), 4),
            "target_mean": round(float(np.nanmean(tar_vals)), 4),
        })

    # Sort so most drifted features appear first
    feature_reports.sort(key=lambda x: x["psi"], reverse=True)

    # Determine overall dataset drift status
    total_feats = max(len(feature_reports), 1)
    if high_drift_count > 2 or (high_drift_count > 0 and high_drift_count / total_feats > 0.15):
        overall_status = "CRITICAL"
    elif high_drift_count > 0 or moderate_drift_count / total_feats > 0.20:
        overall_status = "WARNING"
    else:
        overall_status = "HEALTHY"

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "overall_status": overall_status,
        "total_features_monitored": len(feature_reports),
        "stable_count": len(feature_reports) - high_drift_count - moderate_drift_count,
        "moderate_drift_count": moderate_drift_count,
        "high_drift_count": high_drift_count,
        "features": feature_reports,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Drift analysis complete: overall status={overall_status} "
          f"({summary['stable_count']} STABLE, {moderate_drift_count} MODERATE, {high_drift_count} HIGH)")

    return summary
