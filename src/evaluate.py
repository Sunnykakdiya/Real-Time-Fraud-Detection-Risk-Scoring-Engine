"""
Evaluation, Calibration Analysis & Cost-Matrix Threshold Optimizer.

Calculates PR-AUC, ROC-AUC, Brier score, and optimizes decision thresholds
using a fintech business cost matrix:
  - Missed fraud (FN) cost: 1.0 * TransactionAmt (chargeback loss)
  - False positive (FP) friction: $5.00 (manual review & customer friction)
  - Blocked fraud (TP) cost: $0.00 (loss prevented)
  - True negative (TN) cost: $0.00
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    amounts: np.ndarray | None = None,
) -> Dict[str, float]:
    """Calculate key classification and business performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(threshold, 3),
    }

    if amounts is not None:
        cost_info = compute_cost_and_savings(y_true, y_prob, threshold, amounts)
        metrics.update(cost_info)

    return metrics


def compute_cost_and_savings(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    fp_cost: float = 5.0,
    fn_multiplier: float = 1.0,
) -> Dict[str, float]:
    """
    Compute total financial cost under business cost matrix:
      - FN cost = fn_multiplier * TransactionAmt
      - FP cost = fp_cost (fixed review friction)
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Missed fraud (FN): true fraud, predicted approve (0)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_cost = float(np.sum(amounts[fn_mask] * fn_multiplier))

    # False alarm (FP): legitimate transaction, predicted fraud/review (1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_cost_total = float(np.sum(fp_mask) * fp_cost)

    policy_cost = fn_cost + fp_cost_total

    # Baseline cost: Approve everything (all frauds missed, 0 FPs)
    baseline_approve_all_cost = float(np.sum(amounts[y_true == 1] * fn_multiplier))
    net_savings = baseline_approve_all_cost - policy_cost
    savings_pct = (net_savings / max(baseline_approve_all_cost, 1.0)) * 100.0

    return {
        "policy_cost": round(policy_cost, 2),
        "baseline_cost": round(baseline_approve_all_cost, 2),
        "net_cost_savings": round(net_savings, 2),
        "savings_pct": round(savings_pct, 2),
        "fn_count": int(np.sum(fn_mask)),
        "fp_count": int(np.sum(fp_mask)),
        "tp_count": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn_count": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 5.0,
    fn_multiplier: float = 1.0,
    output_path: str = "models/threshold_config.json",
) -> Dict[str, Any]:
    """
    Grid search to find the optimal decision threshold that maximizes net cost savings.
    Also defines 3 operational risk tiers (LOW/APPROVE, MEDIUM/REVIEW, HIGH/BLOCK).
    """
    threshold_grid = np.linspace(0.02, 0.98, 97)
    best_savings = -float("inf")
    best_thresh = 0.5
    best_metrics: Dict[str, float] = {}

    for t in threshold_grid:
        res = compute_cost_and_savings(y_true, y_prob, t, amounts, fp_cost, fn_multiplier)
        if res["net_cost_savings"] > best_savings:
            best_savings = res["net_cost_savings"]
            best_thresh = float(t)
            best_metrics = res

    # Establish review threshold strictly lower than block threshold
    block_threshold = round(best_thresh, 3)
    review_threshold = round(max(0.005, block_threshold * 0.45), 3)

    threshold_config = {
        "block_threshold": block_threshold,
        "review_threshold": review_threshold,
        "fp_review_cost": fp_cost,
        "fn_multiplier": fn_multiplier,
        "optimal_net_savings": round(best_savings, 2),
        "savings_percentage": best_metrics.get("savings_pct", 0.0),
        "action_tiers": {
            "LOW": {
                "action": "APPROVE",
                "max_score": review_threshold,
                "description": "Legitimate transaction. Automatically cleared without customer friction.",
            },
            "MEDIUM": {
                "action": "REVIEW",
                "min_score": review_threshold,
                "max_score": block_threshold,
                "description": "Moderate risk signals. Routed to 3D-Secure challenge or manual analyst review.",
            },
            "HIGH": {
                "action": "BLOCK",
                "min_score": block_threshold,
                "description": "Critical risk score exceeding cost-optimal loss threshold. Payment declined.",
            },
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(threshold_config, f, indent=2)

    print(f"[OK] Cost-matrix threshold optimized:")
    print(f"     Optimal Block Threshold: {block_threshold} (Net Savings: ${best_savings:,.2f})")
    print(f"     Review Threshold: {review_threshold}")
    print(f"     Configuration saved to {out_file}")

    return threshold_config
