"""
Synthetic Data Generator matching IEEE-CIS Fraud Detection Schema.

Generates realistic fallback datasets (train_transaction.csv and train_identity.csv)
with chronological transaction sequences, authentic class imbalance (~3.5% fraud),
and realistic behavioral signals so the entire pipeline runs out-of-the-box.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_ieee_cis_synthetic_data(
    n_samples: int = 25000,
    seed: int = 42,
    output_dir: str = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic transaction and identity tables matching IEEE-CIS schema."""
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {n_samples:,} synthetic IEEE-CIS transactions (seed={seed})...")

    start_tx_id = 2987000
    transaction_ids = np.arange(start_tx_id, start_tx_id + n_samples)

    # Chronological timestamps: seconds starting from day 1, strictly monotonic or clustered
    # Average gap of ~120 seconds between transactions with occasional bursts
    time_deltas = np.random.exponential(scale=120.0, size=n_samples).astype(np.int64)
    time_deltas = np.clip(time_deltas, 2, 3600)
    transaction_dt = 86400 + np.cumsum(time_deltas)

    # Card entities (simulate ~3,500 recurring cards with power-law activity)
    n_cards = max(100, int(n_samples * 0.15))
    card_pool = np.random.randint(1000, 20000, size=n_cards)
    card_weights = 1.0 / np.arange(1, n_cards + 1) ** 0.8
    card_weights /= card_weights.sum()
    card1 = np.random.choice(card_pool, size=n_samples, p=card_weights)

    card2 = np.random.choice([111.0, 174.0, 264.0, 321.0, 360.0, 490.0, 555.0, np.nan], size=n_samples, p=[0.18, 0.15, 0.15, 0.15, 0.12, 0.12, 0.10, 0.03])
    card3 = np.random.choice([150.0, 185.0, 106.0, np.nan], size=n_samples, p=[0.85, 0.10, 0.03, 0.02])
    card4 = np.random.choice(["visa", "mastercard", "american express", "discover"], size=n_samples, p=[0.65, 0.28, 0.05, 0.02])
    card5 = np.random.choice([102.0, 117.0, 126.0, 137.0, 166.0, 224.0, 226.0, np.nan], size=n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.2, 0.05])
    card6 = np.random.choice(["debit", "credit"], size=n_samples, p=[0.74, 0.26])

    # Product code
    product_cds = ["W", "C", "R", "H", "S"]
    product_cd = np.random.choice(product_cds, size=n_samples, p=[0.75, 0.12, 0.06, 0.05, 0.02])

    # Amounts: lognormal distribution with mean ~$125, realistic decimals (.00, .50, .95, .99)
    base_amounts = np.random.lognormal(mean=4.2, sigma=1.05, size=n_samples)
    decimal_choices = np.random.choice([0.00, 0.50, 0.95, 0.99, 0.25, 0.75], size=n_samples, p=[0.40, 0.20, 0.15, 0.15, 0.05, 0.05])
    transaction_amt = np.round(np.floor(base_amounts) + decimal_choices, 2)
    transaction_amt = np.clip(transaction_amt, 1.0, 8500.0)

    # Email domains
    email_pool = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "comcast.net", "icloud.com", "outlook.com", "protonmail.com"]
    p_email = np.random.choice(email_pool + [None], size=n_samples, p=[0.40, 0.20, 0.12, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.08])
    
    # R_emaildomain: in IEEE-CIS, often NaN for W products, present for C/R
    r_email = []
    for p_prod, p_em in zip(product_cd, p_email):
        if p_prod in ["C", "R"] and np.random.rand() > 0.35:
            # high risk if mismatch
            r_email.append(np.random.choice(email_pool) if np.random.rand() > 0.4 else p_em)
        elif np.random.rand() < 0.10:
            r_email.append(p_em)
        else:
            r_email.append(None)
    r_email = np.array(r_email, dtype=object)

    # Address and distance
    addr1 = np.random.choice(np.arange(100, 500), size=n_samples).astype(np.float32)
    # 10% missing
    addr1[np.random.rand(n_samples) < 0.10] = np.nan
    addr2 = np.random.choice([87.0, 60.0, 96.0, np.nan], size=n_samples, p=[0.88, 0.04, 0.01, 0.07])
    dist1 = np.random.exponential(scale=35.0, size=n_samples)
    dist1[np.random.rand(n_samples) < 0.58] = np.nan

    # Core Count (C), Delta (D), and Vesta (V) features
    c1 = np.random.poisson(lam=1.5, size=n_samples).astype(np.float32)
    c2 = np.random.poisson(lam=1.8, size=n_samples).astype(np.float32)
    c5 = np.random.poisson(lam=1.1, size=n_samples).astype(np.float32)
    c11 = np.random.poisson(lam=1.2, size=n_samples).astype(np.float32)
    c13 = np.random.poisson(lam=2.5, size=n_samples).astype(np.float32)
    c14 = np.random.poisson(lam=1.3, size=n_samples).astype(np.float32)

    d1 = np.random.exponential(scale=45.0, size=n_samples).astype(np.float32)
    d2 = np.random.exponential(scale=60.0, size=n_samples).astype(np.float32)
    d4 = np.random.exponential(scale=80.0, size=n_samples).astype(np.float32)
    d10 = np.random.exponential(scale=30.0, size=n_samples).astype(np.float32)
    d15 = np.random.exponential(scale=120.0, size=n_samples).astype(np.float32)

    # Vesta features subset
    v12 = np.random.choice([0.0, 1.0, 2.0, np.nan], size=n_samples, p=[0.75, 0.18, 0.02, 0.05])
    v34 = np.random.choice([0.0, 1.0, 2.0, np.nan], size=n_samples, p=[0.70, 0.20, 0.05, 0.05])
    v70 = np.random.choice([0.0, 1.0, 2.0, 3.0, np.nan], size=n_samples, p=[0.60, 0.25, 0.08, 0.02, 0.05])
    v91 = np.random.choice([0.0, 1.0, 2.0, np.nan], size=n_samples, p=[0.65, 0.25, 0.05, 0.05])
    v107 = np.random.choice([1.0, 0.0, np.nan], size=n_samples, p=[0.85, 0.10, 0.05])
    v283 = np.random.poisson(lam=0.8, size=n_samples).astype(np.float32)
    v285 = np.random.poisson(lam=0.6, size=n_samples).astype(np.float32)
    v308 = (transaction_amt * np.random.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.5, 0.4, 0.1])).astype(np.float32)

    # Synthesize realistic fraud probability latent score
    # Base fraud rate ~ 3.5%
    hours = (transaction_dt // 3600) % 24
    
    # Latent log-odds of fraud
    latent_fraud_score = -4.2  # baseline ~1.5%
    latent_fraud_score += (transaction_amt > 500) * 1.5
    latent_fraud_score += (transaction_amt > 1500) * 1.2
    latent_fraud_score += (np.isin(hours, [1, 2, 3, 4])) * 0.9  # Night activity
    latent_fraud_score += (product_cd == "C") * 1.3
    latent_fraud_score += (product_cd == "R") * 0.8
    latent_fraud_score += (card6 == "credit") * 0.4
    latent_fraud_score += (p_email == "anonymous.com") * 1.6
    latent_fraud_score += (p_email == "protonmail.com") * 1.4
    latent_fraud_score += (p_email == "mail.com") * 1.1
    
    # Email mismatch penalty
    mismatch_mask = (p_email != r_email) & pd.notna(p_email) & pd.notna(r_email)
    latent_fraud_score += mismatch_mask * 1.8
    
    # High velocity / count flags
    latent_fraud_score += (c1 > 4) * 0.8
    latent_fraud_score += (c13 > 6) * 0.9
    latent_fraud_score += (v283 > 3) * 1.0

    # Convert log-odds to probability
    prob_fraud = 1.0 / (1.0 + np.exp(-latent_fraud_score))
    # Calibrate overall mean to ~3.5%
    prob_fraud = np.clip(prob_fraud, 0.001, 0.95)
    is_fraud = (np.random.rand(n_samples) < prob_fraud).astype(np.int32)
    
    actual_fraud_rate = float(is_fraud.mean() * 100)
    print(f"[*] Synthetic dataset generated: {is_fraud.sum():,} fraud txns ({actual_fraud_rate:.2f}% fraud rate)")

    # Construct train_transaction DataFrame
    df_tx = pd.DataFrame({
        "TransactionID": transaction_ids,
        "isFraud": is_fraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "addr1": addr1,
        "addr2": addr2,
        "dist1": dist1,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "C1": c1,
        "C2": c2,
        "C5": c5,
        "C11": c11,
        "C13": c13,
        "C14": c14,
        "D1": d1,
        "D2": d2,
        "D4": d4,
        "D10": d10,
        "D15": d15,
        "V12": v12,
        "V34": v34,
        "V70": v70,
        "V91": v91,
        "V107": v107,
        "V283": v283,
        "V285": v285,
        "V308": v308,
    })

    # Construct train_identity DataFrame
    # In IEEE-CIS, only ~30% of transactions have identity records, enriched for C and R products and fraud
    identity_prob = np.where(df_tx["ProductCD"].isin(["C", "R", "H"]), 0.70, 0.20)
    identity_prob = np.where(df_tx["isFraud"] == 1, 0.65, identity_prob)
    has_identity = np.random.rand(n_samples) < identity_prob
    
    id_sub_ids = transaction_ids[has_identity]
    n_id = len(id_sub_ids)

    device_types = np.random.choice(["desktop", "mobile"], size=n_id, p=[0.58, 0.42])
    device_infos = []
    id_30_list = []
    id_31_list = []
    
    os_desktop = ["Windows 10", "Mac OS X 10_15_7", "Linux", "Windows 7"]
    os_mobile = ["iOS 13.3.1", "Android 10", "iOS 12.4.1", "Android 9"]
    browser_desktop = ["chrome 80.0", "safari 13.0", "firefox 72.0", "edge 18.0"]
    browser_mobile = ["mobile safari 13.0", "chrome 80.0 for android", "samsung browser 10.1"]

    for dtype in device_types:
        if dtype == "desktop":
            device_infos.append(np.random.choice(["Windows", "MacOS", "Trident/7.0"]))
            id_30_list.append(np.random.choice(os_desktop))
            id_31_list.append(np.random.choice(browser_desktop))
        else:
            device_infos.append(np.random.choice(["iOS Device", "SM-G960F", "Moto G(7)", "SM-A505F"]))
            id_30_list.append(np.random.choice(os_mobile))
            id_31_list.append(np.random.choice(browser_mobile))

    df_id = pd.DataFrame({
        "TransactionID": id_sub_ids,
        "id_01": np.random.uniform(-50.0, 0.0, size=n_id).astype(np.float32),
        "id_02": np.random.randint(1000, 700000, size=n_id).astype(np.float32),
        "id_05": np.random.choice([0.0, 1.0, 2.0, 5.0, np.nan], size=n_id, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        "id_06": np.random.choice([0.0, -5.0, -10.0, np.nan], size=n_id, p=[0.7, 0.1, 0.1, 0.1]),
        "id_11": np.random.choice([100.0, 95.0, 90.0, np.nan], size=n_id, p=[0.8, 0.1, 0.05, 0.05]),
        "id_12": np.random.choice(["Found", "NotFound"], size=n_id, p=[0.6, 0.4]),
        "id_30": id_30_list,
        "id_31": id_31_list,
        "DeviceType": device_types,
        "DeviceInfo": device_infos,
    })

    # Save CSVs
    tx_file = out_path / "train_transaction.csv"
    id_file = out_path / "train_identity.csv"
    
    df_tx.to_csv(tx_file, index=False)
    df_id.to_csv(id_file, index=False)
    print(f"[OK] Saved {len(df_tx):,} rows to {tx_file}")
    print(f"[OK] Saved {len(df_id):,} rows to {id_file}")
    
    return df_tx, df_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IEEE-CIS dataset")
    parser.add_argument("--n-samples", type=int, default=25000, help="Number of transaction rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    generate_ieee_cis_synthetic_data(
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
