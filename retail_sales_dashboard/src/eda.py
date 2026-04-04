"""
Exploratory Data Analysis (EDA) for retail sales.

Computes KPIs, breakdowns, plots, and prints business-style insights.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"


def compute_kpis(df: pd.DataFrame) -> dict:
    """Total revenue (sales), total profit, total units sold."""
    return {
        "total_revenue": float(df["sales_amount"].sum()),
        "total_profit": float(df["profit"].sum()),
        "total_quantity": int(df["quantity"].sum()),
    }


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales_amount by year-month."""
    d = df.copy()
    d["year_month"] = d["date"].dt.to_period("M").astype(str)
    return (
        d.groupby("year_month", as_index=False)["sales_amount"]
        .sum()
        .rename(columns={"sales_amount": "monthly_sales"})
    )


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
        .rename(columns={"sales_amount": "total_sales"})
    )


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region", as_index=False).agg(
            total_sales=("sales_amount", "sum"),
            total_profit=("profit", "sum"),
        )
        .sort_values("total_sales", ascending=False)
    )


def top_n_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        df.groupby("product", as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
        .head(n)
        .rename(columns={"sales_amount": "total_revenue"})
    )


def create_eda_plots(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """
    Save:
    - Line chart: sales over time (daily sum)
    - Bar: sales by category
    - Bar: sales by region
    - Heatmap: correlation of numeric columns
    """
    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # --- Daily sales for line chart ---
    dtmp = df.copy()
    dtmp["day"] = dtmp["date"].dt.floor("D")
    daily = (
        dtmp.groupby("day", as_index=False)["sales_amount"]
        .sum()
        .rename(columns={"sales_amount": "daily_sales"})
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["day"], daily["daily_sales"], color="steelblue", linewidth=1.2)
    ax.set_title("Sales Over Time (Daily Total)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_dir / "sales_over_time.png", dpi=120)
    plt.close(fig)

    # --- Category bar chart ---
    cat_df = sales_by_category(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=cat_df,
        x="category",
        y="total_sales",
        hue="category",
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_title("Total Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales ($)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "sales_by_category.png", dpi=120)
    plt.close(fig)

    # --- Region bar chart ---
    reg_df = sales_by_region(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=reg_df,
        x="region",
        y="total_sales",
        hue="region",
        ax=ax,
        palette="muted",
        legend=False,
    )
    ax.set_title("Total Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    fig.savefig(output_dir / "sales_by_region.png", dpi=120)
    plt.close(fig)

    # --- Correlation heatmap (numeric columns only) ---
    numeric = df[["sales_amount", "quantity", "profit"]].copy()
    # Add ordinal time feature for correlation context
    numeric["day_index"] = (df["date"] - df["date"].min()).dt.days

    fig, ax = plt.subplots(figsize=(6, 5))
    corr = numeric.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"EDA plots saved under: {output_dir}")


def print_business_insights(df: pd.DataFrame) -> None:
    """
    Derive at least 5 actionable, analyst-style insights from the data.
    """
    kpis = compute_kpis(df)
    by_cat = sales_by_category(df)
    by_reg = sales_by_region(df)
    monthly = monthly_sales_trend(df)
    top_prod = top_n_products(df, 5)

    best_cat = by_cat.iloc[0]["category"]
    best_reg = by_reg.iloc[0]["region"]
    worst_reg = by_reg.iloc[-1]["region"]

    # Seasonal: compare Q4 average monthly sales vs Q1
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["year_month"]).dt.month
    q4_sales = m[m["month"].isin([10, 11, 12])]["monthly_sales"].mean()
    q1_sales = m[m["month"].isin([1, 2, 3])]["monthly_sales"].mean()
    seasonal_note = (
        "Q4 (Oct–Dec) monthly sales average higher than Q1 (Jan–Mar)."
        if q4_sales > q1_sales
        else "Q1 monthly sales are comparable to or higher than Q4 in this sample."
    )

    print("\n========== Business Insights ==========")
    print(
        f"1. Scale: Total revenue is ${kpis['total_revenue']:,.0f} with "
        f"${kpis['total_profit']:,.0f} profit across {kpis['total_quantity']:,} units sold."
    )
    print(
        f"2. Category focus: '{best_cat}' leads total sales — prioritize inventory "
        "and marketing for this category."
    )
    print(
        f"3. Region focus: '{best_reg}' generates the highest revenue; '{worst_reg}' "
        "is the weakest — consider targeted promotions or distribution review."
    )
    print(
        f"4. Top products: '{top_prod.iloc[0]['product']}' is the #1 product by revenue — "
        "ensure stock levels and bundle opportunities."
    )
    print(f"5. Seasonality: {seasonal_note}")
    print(
        "6. Action: Use monthly trend charts in leadership reviews to align "
        "targets with visible peaks and dips."
    )
    print("=======================================\n")
