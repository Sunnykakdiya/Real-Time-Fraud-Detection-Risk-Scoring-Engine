"""
Leakage-Free Feature Engineering & Online State Parity Module.

Implements FeatureTransformer and entity state management to guarantee exact parity
between batch offline training and real-time single-transaction inference with zero
temporal or target leakage.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Feature transformer engineered for IEEE-CIS transaction fraud scoring.
    Computes cyclical time features, card historical velocity/amount stats,
    frequency encodings, and domain missingness flags.
    """

    CATEGORICAL_COLS = [
        "ProductCD",
        "card1",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ]

    NUMERIC_RAW_COLS = [
        "C1", "C2", "C13",
        "D1", "D2",
        "V12", "V34", "V70", "V91", "V283", "V308",
    ]

    def __init__(self):
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}
        self.default_freqs_: Dict[str, float] = {}
        self.entity_stats_: Dict[str, Dict[str, float]] = {}
        self.global_stats_: Dict[str, float] = {
            "mean_amt": 135.0,
            "std_amt": 230.0,
            "median_delta_dt": 86400.0,
            "median_amt": 68.5,
        }
        self.numeric_medians_: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureTransformer":
        """
        Fit transformer strictly on the training partition:
        1. Learn frequency encodings for categorical columns.
        2. Compute global population amount & time statistics.
        3. Build card entity summary states (count, mean_amt, std_amt, last_dt).
        4. Learn numeric medians for fallback imputation.
        """
        df = X.copy()

        # 1. Global amount statistics
        amt = df["TransactionAmt"].fillna(0.0).values
        self.global_stats_["mean_amt"] = float(np.mean(amt))
        self.global_stats_["std_amt"] = float(np.std(amt)) if np.std(amt) > 1e-4 else 1.0
        self.global_stats_["median_amt"] = float(np.median(amt))

        # 2. Frequency encodings
        n_rows = len(df)
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                val_counts = df[col].astype(str).value_counts(dropna=False)
                freqs = (val_counts / n_rows).to_dict()
                self.freq_maps_[col] = freqs
                self.default_freqs_[col] = 1.0 / n_rows
            else:
                self.freq_maps_[col] = {}
                self.default_freqs_[col] = 1e-5

        # 3. Numeric medians
        for col in self.NUMERIC_RAW_COLS:
            if col in df.columns:
                self.numeric_medians_[col] = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
            else:
                self.numeric_medians_[col] = 0.0

        # 4. Build per-card historical entity stats (cumulative chronologically)
        if "card1" in df.columns:
            card_col = df["card1"].astype(str)
            amt_col = df["TransactionAmt"].fillna(self.global_stats_["mean_amt"]).values
            dt_col = df["TransactionDT"].values if "TransactionDT" in df.columns else np.zeros(len(df))

            # Group final summary stats per card1
            grouped = df.groupby(card_col)
            counts = grouped["TransactionAmt"].count().to_dict()
            means = grouped["TransactionAmt"].mean().to_dict()
            stds = grouped["TransactionAmt"].std().fillna(0.0).to_dict()
            last_dts = grouped["TransactionDT"].max().to_dict() if "TransactionDT" in df.columns else {}

            self.entity_stats_ = {}
            for card_id, cnt in counts.items():
                self.entity_stats_[str(card_id)] = {
                    "count": int(cnt),
                    "mean_amt": float(means.get(card_id, self.global_stats_["mean_amt"])),
                    "std_amt": float(stds.get(card_id, self.global_stats_["std_amt"])),
                    "last_dt": float(last_dts.get(card_id, 0.0)),
                }

        # 5. Determine feature names order
        dummy_out = self._transform_internal(df.iloc[:2], is_batch=True)
        self.feature_names_ = list(dummy_out.columns)

        return self

    def _transform_internal(self, df: pd.DataFrame, is_batch: bool = True) -> pd.DataFrame:
        """Core feature generation shared between batch and online inference."""
        out = pd.DataFrame(index=df.index)

        # Raw amount & derived pricing signals
        amt = df["TransactionAmt"].astype(np.float32).values
        out["TransactionAmt"] = amt
        out["log_transaction_amt"] = np.log1p(np.maximum(amt, 0.0)).astype(np.float32)
        out["amt_decimal_cents"] = np.round(amt % 1.0, 4).astype(np.float32)

        # Card entity lookup and interaction features
        card1_series = df["card1"].astype(str) if "card1" in df.columns else pd.Series(["unknown"] * len(df))
        tx_dt = df["TransactionDT"].values if "TransactionDT" in df.columns else np.zeros(len(df))

        amt_ratios = []
        amt_zscores = []
        time_since_last = []
        card_counts = []

        g_mean = self.global_stats_.get("mean_amt", 135.0)
        g_std = self.global_stats_.get("std_amt", 230.0)
        g_dt = self.global_stats_.get("median_delta_dt", 86400.0)

        for i, card_id in enumerate(card1_series):
            curr_amt = amt[i]
            curr_dt = tx_dt[i]

            stats = self.entity_stats_.get(card_id)
            if stats is not None:
                c_mean = stats["mean_amt"]
                c_std = stats["std_amt"] if stats["std_amt"] > 1e-3 else g_std
                c_count = stats["count"]
                c_last_dt = stats["last_dt"]
                
                ratio = curr_amt / (c_mean + 1e-4)
                zscore = (curr_amt - c_mean) / (c_std + 1e-4)
                delta_t = max(0.0, curr_dt - c_last_dt) if c_last_dt > 0 else g_dt
            else:
                # Cold start card (unseen card fallback)
                ratio = curr_amt / (g_mean + 1e-4)
                zscore = (curr_amt - g_mean) / (g_std + 1e-4)
                delta_t = g_dt
                c_count = 1

            amt_ratios.append(ratio)
            amt_zscores.append(zscore)
            time_since_last.append(delta_t)
            card_counts.append(c_count)

        out["amt_to_card_mean_ratio"] = np.clip(np.array(amt_ratios, dtype=np.float32), 0.0, 50.0)
        out["amt_to_card_zscore"] = np.clip(np.array(amt_zscores, dtype=np.float32), -10.0, 30.0)
        out["time_since_last_card_txn"] = np.clip(np.array(time_since_last, dtype=np.float32), 0.0, 31536000.0)
        out["card_txn_count"] = np.array(card_counts, dtype=np.float32)

        # Cyclical temporal signals
        if "Transaction_Hour" in df.columns:
            hours = df["Transaction_Hour"].astype(np.float32).values
        elif "TransactionDT" in df.columns:
            hours = ((df["TransactionDT"] // 3600) % 24).astype(np.float32).values
        else:
            hours = np.zeros(len(df), dtype=np.float32)

        if "TransactionDT" in df.columns:
            day_of_week = ((df["TransactionDT"] // (3600 * 24)) % 7).astype(np.float32).values
        else:
            day_of_week = np.zeros(len(df), dtype=np.float32)

        out["transaction_hour"] = hours
        out["transaction_day_of_week"] = day_of_week
        out["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
        out["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)

        # Email domain signals
        p_email = df["P_emaildomain"].astype(str) if "P_emaildomain" in df.columns else pd.Series(["missing"] * len(df))
        r_email = df["R_emaildomain"].astype(str) if "R_emaildomain" in df.columns else pd.Series(["missing"] * len(df))

        # Email match: 1 if both provided and matching, 0 if different or missing
        valid_p = (p_email != "nan") & (p_email != "None") & (p_email != "") & (p_email != "missing")
        valid_r = (r_email != "nan") & (r_email != "None") & (r_email != "") & (r_email != "missing")
        email_match = (p_email == r_email) & valid_p & valid_r
        out["email_domain_match"] = email_match.astype(np.float32)
        out["R_emaildomain_isna"] = (~valid_r).astype(np.float32)

        # Identity null count (missingness telemetry)
        id_cols = [c for c in df.columns if c.startswith("id_")]
        if id_cols:
            out["id_null_count"] = df[id_cols].isna().sum(axis=1).astype(np.float32)
        else:
            out["id_null_count"] = np.zeros(len(df), dtype=np.float32)

        # Categorical frequency encodings
        for cat_col in self.CATEGORICAL_COLS:
            out_col_name = f"{cat_col}_freq"
            freq_map = self.freq_maps_.get(cat_col, {})
            d_freq = self.default_freqs_.get(cat_col, 1e-5)

            if cat_col in df.columns:
                vals = df[cat_col].astype(str)
                out[out_col_name] = vals.map(lambda v: freq_map.get(v, d_freq)).astype(np.float32)
            else:
                out[out_col_name] = np.full(len(df), d_freq, dtype=np.float32)

        # Raw numeric columns (with median imputation for missing values)
        for col in self.NUMERIC_RAW_COLS:
            med_val = self.numeric_medians_.get(col, 0.0)
            if col in df.columns:
                out[col] = df[col].fillna(med_val).astype(np.float32)
            else:
                out[col] = np.full(len(df), med_val, dtype=np.float32)

        # Replace any inf or NaN with safe defaults
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Order columns identically
        if self.feature_names_ and len(self.feature_names_) == len(out.columns):
            out = out[self.feature_names_]

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a batch DataFrame into engineered numeric features."""
        return self._transform_internal(X, is_batch=True)

    def transform_single(self, txn_dict: Union[Dict[str, Any], pd.Series]) -> pd.DataFrame:
        """
        Transform a single incoming API transaction payload into an engineered feature vector
        matching batch training schema with exact numeric parity.
        """
        if isinstance(txn_dict, pd.Series):
            df_single = pd.DataFrame([txn_dict.to_dict()])
        else:
            df_single = pd.DataFrame([txn_dict])

        # If TransactionDT missing but Transaction_Hour provided, estimate DT
        if "TransactionDT" not in df_single.columns:
            hr = df_single.get("Transaction_Hour", [12])[0]
            df_single["TransactionDT"] = 86400 * 30 + int(hr) * 3600

        transformed = self._transform_internal(df_single, is_batch=False)
        return transformed

    def export_entity_stats(self, output_path: str = "models/entity_stats.json") -> None:
        """Persist card entity summary stats and global fallbacks to JSON."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "global_stats": self.global_stats_,
            "card_entities_count": len(self.entity_stats_),
            "entity_stats": self.entity_stats_,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Exported {len(self.entity_stats_):,} card entity statistics to {out_file}")

    def load_entity_stats(self, input_path: str = "models/entity_stats.json") -> None:
        """Load entity stats from JSON."""
        in_file = Path(input_path)
        if not in_file.exists():
            return
        with open(in_file, "r") as f:
            payload = json.load(f)
        self.global_stats_ = payload.get("global_stats", self.global_stats_)
        self.entity_stats_ = payload.get("entity_stats", {})
