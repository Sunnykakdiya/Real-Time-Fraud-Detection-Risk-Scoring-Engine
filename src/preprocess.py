"""
Preprocessing and Memory Optimization Module.

Performs table merge between transaction and identity tables, downcasts numeric dtypes
to reduce RAM footprint on 16GB machines, prunes columns with >85% missingness or zero
variance, and caches high-performance Parquet artifacts in data/processed/.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def downcast_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric types to reduce RAM footprint."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct_reduced = 100 * (start_mem - end_mem) / max(start_mem, 1e-6)
        print(f"[*] Memory downcasted from {start_mem:.2f} MB to {end_mem:.2f} MB (-{pct_reduced:.1f}%)")

    return df


def prune_uninformative_columns(
    df: pd.DataFrame,
    null_threshold: float = 0.85,
    protected_cols: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Prune columns with missingness exceeding threshold or zero variance."""
    if protected_cols is None:
        protected_cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"]

    n_rows = len(df)
    dropped_cols = []

    for col in df.columns:
        if col in protected_cols:
            continue

        # Missingness check
        null_rate = df[col].isna().sum() / n_rows
        if null_rate > null_threshold:
            dropped_cols.append(col)
            continue

        # Zero variance check for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0 and valid_vals.nunique() <= 1:
                dropped_cols.append(col)
                continue

    pruned_df = df.drop(columns=dropped_cols)
    if verbose and dropped_cols:
        print(f"[*] Pruned {len(dropped_cols)} uninformative/high-null columns: {dropped_cols[:5]}...")

    return pruned_df, dropped_cols


def preprocess_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    null_threshold: float = 0.85,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge transaction and identity tables, downcast memory, prune uninformative cols, and export Parquet."""
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    tx_file = raw_path / "train_transaction.csv"
    id_file = raw_path / "train_identity.csv"

    if not tx_file.exists():
        raise FileNotFoundError(f"Missing {tx_file}. Run generate_sample_data.py or download Kaggle files.")

    if verbose:
        print(f"[*] Loading raw tables from {raw_path}...")

    df_tx = pd.read_csv(tx_file)
    if id_file.exists():
        df_id = pd.read_csv(id_file)
        if verbose:
            print(f"[*] Merging transaction ({len(df_tx):,} rows) and identity ({len(df_id):,} rows)...")
        df_merged = pd.merge(df_tx, df_id, on="TransactionID", how="left")
    else:
        df_merged = df_tx

    # Downcast numeric dtypes
    df_merged = downcast_dtypes(df_merged, verbose=verbose)

    # Prune high-null or zero-variance columns
    df_merged, _ = prune_uninformative_columns(df_merged, null_threshold=null_threshold, verbose=verbose)

    # Sort strictly chronologically by TransactionDT
    if "TransactionDT" in df_merged.columns:
        df_merged = df_merged.sort_values("TransactionDT").reset_index(drop=True)

    # Export to Parquet
    out_parquet = proc_path / "train_merged.parquet"
    df_merged.to_parquet(out_parquet, index=False, engine="pyarrow")

    if verbose:
        print(f"[OK] Preprocessed data saved to {out_parquet} ({len(df_merged):,} rows, {len(df_merged.columns)} columns)")

    return df_merged


def load_processed_data(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Load cached preprocessed Parquet dataset."""
    parquet_path = Path(processed_dir) / "train_merged.parquet"
    if not parquet_path.exists():
        print(f"[!] {parquet_path} not found. Running preprocess_data()...")
        return preprocess_data(processed_dir=processed_dir)
    return pd.read_parquet(parquet_path, engine="pyarrow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and downcast IEEE-CIS datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory with raw CSV files")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory to save Parquet")
    parser.add_argument("--null-threshold", type=float, default=0.85, help="Drop columns with null rate > threshold")
    args = parser.parse_args()

    preprocess_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        null_threshold=args.null_threshold,
    )
