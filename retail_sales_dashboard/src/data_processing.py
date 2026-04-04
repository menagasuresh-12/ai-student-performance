"""
Retail Sales Dashboard — data loading, synthetic generation, and cleaning.

This module handles:
- Creating a realistic synthetic retail dataset if no CSV exists
- Loading CSV with Pandas
- Cleaning: missing values, duplicates, date parsing, dtypes
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (project root = parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "retail_sales_raw.csv"
CLEAN_CSV = DATA_DIR / "retail_sales_cleaned.csv"

# Product catalog for synthetic data (realistic retail-style names)
CATEGORIES = {
    "Electronics": ["USB Hub", "Wireless Mouse", "Bluetooth Speaker", "Phone Case"],
    "Furniture": ["Office Chair", "Desk Lamp", "Bookshelf", "Filing Cabinet"],
    "Office Supplies": ["Notebook Set", "Pen Pack", "Stapler", "Binder"],
    "Apparel": ["Polo Shirt", "Winter Jacket", "Running Shoes", "Baseball Cap"],
}
REGIONS = ["North", "South", "East", "West", "Central"]


def generate_synthetic_retail_data(
    n_rows: int = 5000,
    start_date: str = "2022-01-01",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic retail dataset with columns:
    date, product, category, region, sales_amount, quantity, profit

    We add mild seasonality and weekend effects so trends look realistic.
    """
    rng = np.random.default_rng(random_seed)

    # Build a flat list of (product, category) pairs
    product_rows = []
    for category, products in CATEGORIES.items():
        for p in products:
            product_rows.append((p, category))

    rows = []
    start_ts = pd.Timestamp(start_date)
    for i in range(n_rows):
        # Pick a random calendar day in a 2-year window for variety
        day_offset = int(rng.integers(0, 730))
        base = start_ts + pd.Timedelta(days=day_offset)
        # Mild upward revenue trend over time (helps the linear forecast lesson)
        trend = 1.0 + (day_offset / 730.0) * 0.25
        # Slight uplift in November–December (holiday season)
        month = base.month
        seasonal = 1.15 if month in (11, 12) else 1.0
        if month in (6, 7):
            seasonal *= 1.05  # small summer bump

        prod, cat = product_rows[rng.integers(0, len(product_rows))]
        region = rng.choice(REGIONS)
        qty = int(rng.integers(1, 15))
        unit_price = float(rng.uniform(8, 120))
        sales_amount = round(
            unit_price * qty * seasonal * trend * rng.uniform(0.85, 1.15), 2
        )
        margin = float(rng.uniform(0.08, 0.35))
        profit = round(sales_amount * margin, 2)

        rows.append(
            {
                "date": base.strftime("%Y-%m-%d"),
                "product": prod,
                "category": cat,
                "region": region,
                "sales_amount": sales_amount,
                "quantity": qty,
                "profit": profit,
            }
        )

    df = pd.DataFrame(rows)
    # Shuffle so rows are not sorted by date
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Inject some missing values for teaching (small fraction)
    miss_idx = rng.choice(df.index, size=max(1, n_rows // 200), replace=False)
    df.loc[miss_idx[: len(miss_idx) // 3], "region"] = np.nan
    df.loc[miss_idx[len(miss_idx) // 3 : 2 * len(miss_idx) // 3], "profit"] = np.nan

    return df


def load_or_create_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Load retail_sales_raw.csv if it exists; otherwise generate synthetic data and save it.
    """
    path = path or RAW_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Loading existing raw dataset: {path}")
        df = pd.read_csv(path)
    else:
        print("No raw CSV found. Generating synthetic retail data...")
        df = generate_synthetic_retail_data(n_rows=5000)
        df.to_csv(path, index=False)
        print(f"Saved synthetic raw data to: {path}")

    return df


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the retail dataframe:
    - Parse date column to datetime
    - Drop duplicate rows
    - Fix missing values (region -> 'Unknown', profit imputed from margin proxy)
    - Ensure numeric dtypes
    """
    df = df.copy()

    # --- Parse dates ---
    if "date" not in df.columns:
        raise ValueError("Expected a column named 'date'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Remove full duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    # --- Missing dates: drop rows with invalid date ---
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        print(f"Dropping {invalid_dates} rows with invalid dates.")
    df = df.dropna(subset=["date"])

    # --- Region: fill missing with 'Unknown' ---
    df["region"] = df["region"].fillna("Unknown")

    # --- Profit: if missing, estimate from median profit ratio where possible ---
    med_ratio = (df["profit"] / df["sales_amount"]).median()
    if pd.isna(med_ratio) or med_ratio <= 0:
        med_ratio = 0.2
    mask_profit_na = df["profit"].isna()
    df.loc[mask_profit_na, "profit"] = (
        df.loc[mask_profit_na, "sales_amount"] * med_ratio
    ).round(2)

    # --- Quantity and sales: coerce to numeric, drop bad rows ---
    for col in ["sales_amount", "quantity", "profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["sales_amount", "quantity", "profit"])
    df["quantity"] = df["quantity"].astype(int)

    # --- Text columns as string ---
    for col in ["product", "category", "region"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def explain_columns() -> None:
    """Print beginner-friendly descriptions of each column."""
    print("\n--- Column meanings ---")
    print("date          : Day the sale was recorded.")
    print("product       : Name of the item sold.")
    print("category      : High-level product group (e.g. Electronics).")
    print("region        : Sales region (geographic bucket).")
    print("sales_amount  : Total money from the line (price × quantity style).")
    print("quantity      : Number of units sold.")
    print("profit        : Estimated profit dollars for that line.")
