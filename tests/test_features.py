"""
Unit Tests for Leakage-Free Feature Engineering & Online State Parity.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FeatureTransformer


@pytest.fixture
def sample_training_data():
    """Create deterministic sample transaction sequence for testing."""
    df = pd.DataFrame({
        "TransactionID": [100, 101, 102, 103, 104],
        "TransactionDT": [86400, 86500, 90000, 95000, 100000],
        "TransactionAmt": [50.0, 100.0, 150.0, 200.0, 50.0],
        "ProductCD": ["W", "W", "C", "R", "W"],
        "card1": [1001, 1001, 1002, 1001, 1003],
        "card4": ["visa", "visa", "mastercard", "visa", "discover"],
        "card6": ["debit", "debit", "credit", "debit", "credit"],
        "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "anonymous.com", "gmail.com"],
        "R_emaildomain": [None, None, "yahoo.com", "protonmail.com", None],
        "DeviceType": ["desktop", "desktop", "mobile", "mobile", "desktop"],
        "C1": [1.0, 1.0, 3.0, 5.0, 1.0],
        "C2": [1.0, 1.0, 2.0, 4.0, 1.0],
        "C13": [2.0, 3.0, 5.0, 8.0, 1.0],
        "D1": [10.0, 12.0, 0.0, 45.0, 2.0],
        "D2": [10.0, 12.0, 0.0, 30.0, 2.0],
        "V12": [1.0, 1.0, 0.0, 2.0, 1.0],
        "V34": [0.0, 0.0, 1.0, 1.0, 0.0],
        "V70": [0.0, 1.0, 2.0, 2.0, 0.0],
        "V91": [1.0, 1.0, 0.0, 1.0, 1.0],
        "V283": [0.0, 1.0, 3.0, 4.0, 0.0],
        "V308": [50.0, 150.0, 450.0, 800.0, 50.0],
    })
    return df


def test_fit_and_feature_names(sample_training_data):
    """Verify transformer fits and defines canonical feature set."""
    transformer = FeatureTransformer()
    transformer.fit(sample_training_data)

    assert len(transformer.feature_names_) > 20
    assert "TransactionAmt" in transformer.feature_names_
    assert "amt_to_card_mean_ratio" in transformer.feature_names_
    assert "hour_sin" in transformer.feature_names_
    assert "hour_cos" in transformer.feature_names_
    assert "email_domain_match" in transformer.feature_names_


def test_batch_vs_single_parity(sample_training_data):
    """
    Test critical invariant: transforming 1 row via transform_single produces
    identical features to passing that same row through batch transform.
    """
    transformer = FeatureTransformer()
    transformer.fit(sample_training_data)

    row = sample_training_data.iloc[2:3]
    row_dict = row.iloc[0].to_dict()

    batch_feat = transformer.transform(row)
    single_feat = transformer.transform_single(row_dict)

    # Columns must match exactly in name and order
    assert list(batch_feat.columns) == list(single_feat.columns)

    # Numeric values must match within strict float32 precision
    for col in batch_feat.columns:
        b_val = float(batch_feat[col].iloc[0])
        s_val = float(single_feat[col].iloc[0])
        assert np.isclose(b_val, s_val, atol=1e-4), f"Parity mismatch on {col}: {b_val} != {s_val}"


def test_cold_start_card_handling(sample_training_data):
    """Verify that an unseen card (cold start) falls back gracefully without NaNs or crashes."""
    transformer = FeatureTransformer()
    transformer.fit(sample_training_data)

    unseen_txn = {
        "TransactionAmt": 125.0,
        "ProductCD": "W",
        "card1": 99999,  # Never seen in training data
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "DeviceType": "desktop",
        "Transaction_Hour": 15,
    }

    feat_df = transformer.transform_single(unseen_txn)
    assert not feat_df.isna().any().any(), "Cold start transaction produced NaN features"
    assert not np.isinf(feat_df.values).any(), "Cold start transaction produced Infinite features"

    # Card count should default to 1
    assert feat_df["card_txn_count"].iloc[0] == 1.0
    # Ratio should be positive and finite
    assert feat_df["amt_to_card_mean_ratio"].iloc[0] > 0.0


def test_cyclical_time_trigonometry(sample_training_data):
    """Verify that hour_sin^2 + hour_cos^2 == 1.0 for all hours."""
    transformer = FeatureTransformer()
    transformer.fit(sample_training_data)

    for hr in range(24):
        txn = {"TransactionAmt": 50.0, "Transaction_Hour": hr, "card1": 1001}
        f_df = transformer.transform_single(txn)
        s = f_df["hour_sin"].iloc[0]
        c = f_df["hour_cos"].iloc[0]
        assert np.isclose(s**2 + c**2, 1.0, atol=1e-5), f"Trig identity failed for hour {hr}"


def test_email_matching_logic(sample_training_data):
    """Verify email match flag is 1 when both match and 0 when missing or different."""
    transformer = FeatureTransformer()
    transformer.fit(sample_training_data)

    matching_txn = {
        "TransactionAmt": 50.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
    }
    f_match = transformer.transform_single(matching_txn)
    assert f_match["email_domain_match"].iloc[0] == 1.0

    mismatch_txn = {
        "TransactionAmt": 50.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "yahoo.com",
    }
    f_mismatch = transformer.transform_single(mismatch_txn)
    assert f_mismatch["email_domain_match"].iloc[0] == 0.0

    missing_txn = {
        "TransactionAmt": 50.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
    }
    f_missing = transformer.transform_single(missing_txn)
    assert f_missing["email_domain_match"].iloc[0] == 0.0
