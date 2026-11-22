"""
Model Training, Calibration & MLflow Experiment Tracking Pipeline.

Executes:
1. Strict chronological split (70% train, 15% validation, 15% holdout test).
2. Leakage-free feature engineering fitted on train partition.
3. Unsupervised Isolation Forest anomaly meta-feature generation.
4. Model training & comparison: Logistic Regression, LightGBM, and XGBoost.
5. Probability calibration (CalibratedClassifierCV on validation split).
6. Cost-matrix decision threshold optimization.
7. MLflow experiment logging and production bundle export to models/model_bundle.joblib.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.drift import analyze_drift
from src.evaluate import compute_metrics, optimize_threshold
from src.features import FeatureTransformer
from src.preprocess import load_processed_data


def chronological_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data strictly chronologically by TransactionDT to prevent future leakage.
    No random shuffle, no SMOTE synthetic inflation.
    """
    if "TransactionDT" in df.columns:
        df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()

    print(f"[*] Chronological split:")
    print(f"    Train: {len(df_train):,} rows ({df_train['isFraud'].mean()*100:.2f}% fraud)")
    print(f"    Val:   {len(df_val):,} rows ({df_val['isFraud'].mean()*100:.2f}% fraud)")
    print(f"    Test:  {len(df_test):,} rows ({df_test['isFraud'].mean()*100:.2f}% fraud)")

    return df_train, df_val, df_test


def train_pipeline(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    quick: bool = False,
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db",
) -> Dict[str, Any]:
    """Run full training, calibration, and evaluation pipeline."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    df = load_processed_data(processed_dir=data_dir)

    # 1. Chronological Split
    df_train, df_val, df_test = chronological_split(df, train_pct=0.70, val_pct=0.15)

    y_train = df_train["isFraud"].values.astype(int)
    y_val = df_val["isFraud"].values.astype(int)
    y_test = df_test["isFraud"].values.astype(int)

    amt_train = df_train["TransactionAmt"].values.astype(float)
    amt_val = df_val["TransactionAmt"].values.astype(float)
    amt_test = df_test["TransactionAmt"].values.astype(float)

    # 2. Fit Feature Transformer strictly on Train split
    print("[*] Fitting FeatureTransformer on train partition...")
    transformer = FeatureTransformer()
    transformer.fit(df_train)

    X_train_df = transformer.transform(df_train)
    X_val_df = transformer.transform(df_val)
    X_test_df = transformer.transform(df_test)

    # Export entity stats
    transformer.export_entity_stats(f"{models_dir}/entity_stats.json")

    # 3. Fit Isolation Forest unsupervised meta-feature on Train
    print("[*] Training Isolation Forest anomaly detector on train split...")
    iforest = IsolationForest(
        n_estimators=100 if not quick else 50,
        contamination=float(np.clip(y_train.mean(), 0.01, 0.10)),
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_train_df)

    # Anomaly score: higher means more anomalous (negate decision function)
    X_train_df["iforest_anomaly_score"] = -iforest.decision_function(X_train_df)
    X_val_df["iforest_anomaly_score"] = -iforest.decision_function(X_val_df)
    X_test_df["iforest_anomaly_score"] = -iforest.decision_function(X_test_df)

    # Update transformer feature names to include iforest_anomaly_score
    feature_names = list(X_train_df.columns)
    transformer.feature_names_ = feature_names

    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values

    # Imbalance scale factor for tree models
    pos_count = max(int(np.sum(y_train == 1)), 1)
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / pos_count)

    print(f"[*] Class imbalance ratio: {scale_pos_weight:.1f}x negative to positive")

    # MLflow Setup
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection-engine")

    benchmark_results: Dict[str, Any] = {}

    # 4a. Model 1: Logistic Regression Baseline
    print("\n[*] Training Baseline: Logistic Regression (Imputed + Scaled)...")
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        lr_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=42)),
        ])
        lr_pipe.fit(X_train, y_train)
        lr_val_probs = lr_pipe.predict_proba(X_val)[:, 1]
        lr_test_probs = lr_pipe.predict_proba(X_test)[:, 1]

        lr_metrics = compute_metrics(y_test, lr_test_probs, threshold=0.5, amounts=amt_test)
        mlflow.log_params({"model_type": "LogisticRegression", "C": 0.1, "class_weight": "balanced"})
        mlflow.log_metrics(lr_metrics)
        benchmark_results["Logistic Regression"] = lr_metrics
        print(f"    Logistic Regression -> Test PR-AUC: {lr_metrics['pr_auc']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")

    # 4b. Model 2: XGBoost Classifier
    print("\n[*] Training XGBoost Classifier...")
    with mlflow.start_run(run_name="XGBoost_Classifier"):
        n_estimators_xgb = 120 if quick else 250
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators_xgb,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = compute_metrics(y_test, xgb_test_probs, threshold=0.5, amounts=amt_test)
        mlflow.log_params({"model_type": "XGBoost", "n_estimators": n_estimators_xgb, "scale_pos_weight": round(scale_pos_weight, 2)})
        mlflow.log_metrics(xgb_metrics)
        benchmark_results["XGBoost"] = xgb_metrics
        print(f"    XGBoost -> Test PR-AUC: {xgb_metrics['pr_auc']:.4f}, ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

    # 4c. Model 3: LightGBM Classifier
    print("\n[*] Training LightGBM Classifier...")
    with mlflow.start_run(run_name="LightGBM_Classifier"):
        n_estimators_lgb = 150 if quick else 300
        lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators_lgb,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        lgb_raw_val_probs = lgb_model.predict_proba(X_val)[:, 1]
        lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]

        lgb_metrics = compute_metrics(y_test, lgb_test_probs, threshold=0.5, amounts=amt_test)
        mlflow.log_params({"model_type": "LightGBM", "n_estimators": n_estimators_lgb, "scale_pos_weight": round(scale_pos_weight, 2)})
        mlflow.log_metrics(lgb_metrics)
        benchmark_results["LightGBM"] = lgb_metrics
        print(f"    LightGBM -> Test PR-AUC: {lgb_metrics['pr_auc']:.4f}, ROC-AUC: {lgb_metrics['roc_auc']:.4f}")

    # 5. Probability Calibration (CalibratedClassifierCV on Validation Split)
    print("\n[*] Calibrating probabilities with Sigmoid CalibratedClassifierCV on validation split...")
    with mlflow.start_run(run_name="LightGBM_Calibrated_Production"):
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated_model = CalibratedClassifierCV(
                estimator=FrozenEstimator(lgb_model),
                method="sigmoid",
            )
        except ImportError:
            calibrated_model = CalibratedClassifierCV(
                estimator=lgb_model,
                method="sigmoid",
                cv="prefit",
            )
        calibrated_model.fit(X_val, y_val)

        cal_val_probs = calibrated_model.predict_proba(X_val)[:, 1]
        cal_test_probs = calibrated_model.predict_proba(X_test)[:, 1]

        # 6. Optimize Decision Threshold via Business Cost Matrix
        threshold_config = optimize_threshold(
            y_true=y_val,
            y_prob=cal_val_probs,
            amounts=amt_val,
            fp_cost=5.0,
            fn_multiplier=1.0,
            output_path=f"{models_dir}/threshold_config.json",
        )
        opt_thresh = threshold_config["block_threshold"]

        cal_metrics = compute_metrics(y_test, cal_test_probs, threshold=opt_thresh, amounts=amt_test)
        mlflow.log_params({
            "model_type": "LightGBM_Calibrated",
            "calibration_method": "sigmoid",
            "block_threshold": opt_thresh,
            "review_threshold": threshold_config["review_threshold"],
        })
        mlflow.log_metrics(cal_metrics)
        benchmark_results["LightGBM (Calibrated Production)"] = cal_metrics
        print(f"    Calibrated LightGBM (thresh={opt_thresh}) -> PR-AUC: {cal_metrics['pr_auc']:.4f}, Brier: {cal_metrics['brier_score']:.4f}, Net Savings: ${cal_metrics['net_cost_savings']:,.2f}")

    # 7. Drift Baseline Analysis: Holdout Test vs Train Partition
    print("\n[*] Running initial drift monitoring report (Test vs Train)...")
    drift_report = analyze_drift(
        ref_df=X_train_df,
        target_df=X_test_df,
        features=feature_names[:15],
        output_path=f"{models_dir}/drift_report.json",
    )

    # 8. Save Production Model Bundle
    bundle_path = Path(models_dir) / "model_bundle.joblib"
    bundle = {
        "calibrated_model": calibrated_model,
        "raw_tree_booster": lgb_model,
        "feature_transformer": transformer,
        "isolation_forest": iforest,
        "feature_names": feature_names,
        "best_model_name": "LightGBM (Calibrated)",
        "threshold_config": threshold_config,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train_samples": len(df_train),
        "test_metrics": cal_metrics,
    }
    joblib.dump(bundle, bundle_path)
    print(f"[OK] Production model bundle saved to {bundle_path}")

    # Save benchmark metrics to metrics.json
    metrics_path = Path(models_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "models": benchmark_results,
            "best_model": "LightGBM (Calibrated Production)",
            "primary_metric": "pr_auc",
            "best_pr_auc": cal_metrics["pr_auc"],
            "best_roc_auc": cal_metrics["roc_auc"],
            "brier_score": cal_metrics["brier_score"],
            "net_cost_savings": cal_metrics.get("net_cost_savings", 0.0),
            "savings_pct": cal_metrics.get("savings_pct", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)
    print(f"[OK] Benchmark metrics saved to {metrics_path}")

    return {
        "bundle_path": str(bundle_path),
        "metrics": cal_metrics,
        "benchmark": benchmark_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and calibrate fraud detection models")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--quick", action="store_true", help="Quick training with fewer boosting rounds")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        quick=args.quick,
    )
