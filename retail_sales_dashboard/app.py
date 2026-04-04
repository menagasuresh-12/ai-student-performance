"""
Streamlit dashboard: Retail Sales Analysis and Forecasting.

Run from the `retail_sales_dashboard` folder:

    streamlit run app.py

Shows KPIs, top products, sales trends, and (if available) a simple forecast.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import compute_kpis, monthly_sales_trend, top_n_products
from src import model_forecast as mf

CLEAN_CSV = ROOT / "data" / "retail_sales_cleaned.csv"
MODEL_PATH = ROOT / "models" / "sales_forecast_model.pkl"


@st.cache_data
def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def main() -> None:
    st.set_page_config(
        page_title="Retail Sales Dashboard",
        layout="wide",
    )
    st.title("Retail Sales Analysis and Forecasting Dashboard")
    st.caption("Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Streamlit")

    if not CLEAN_CSV.exists():
        st.error(
            f"Cleaned data not found at `{CLEAN_CSV}`. "
            "Run `python run_pipeline.py` first to generate it."
        )
        return

    df = load_clean_data(CLEAN_CSV)

    # ----- KPI row -----
    kpis = compute_kpis(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total revenue", f"${kpis['total_revenue']:,.0f}")
    c2.metric("Total profit", f"${kpis['total_profit']:,.0f}")
    c3.metric("Units sold", f"{kpis['total_quantity']:,}")

    st.divider()

    # ----- Top products -----
    st.subheader("Top products by revenue")
    top5 = top_n_products(df, 5)
    st.dataframe(top5, use_container_width=True, hide_index=True)

    # ----- Monthly trend (interactive) -----
    st.subheader("Monthly sales trend")
    monthly = monthly_sales_trend(df)
    chart_df = monthly.rename(columns={"year_month": "month", "monthly_sales": "sales"})
    st.line_chart(chart_df.set_index("month")["sales"])

    # ----- Category / region (compact charts) -----
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Sales by category**")
        cat = df.groupby("category", as_index=False)["sales_amount"].sum()
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(
            data=cat,
            x="category",
            y="sales_amount",
            hue="category",
            ax=ax1,
            palette="viridis",
            legend=False,
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha="right")
        ax1.set_ylabel("Sales ($)")
        ax1.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        st.markdown("**Sales by region**")
        reg = df.groupby("region", as_index=False)["sales_amount"].sum()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(
            data=reg,
            x="region",
            y="sales_amount",
            hue="region",
            ax=ax2,
            palette="muted",
            legend=False,
        )
        ax2.set_ylabel("Sales ($)")
        ax2.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ----- Forecast section -----
    st.subheader("Sales forecast (simple linear trend)")
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        daily_feat, _, _ = mf.prepare_forecast_features(mf.daily_sales_series(df))
        future = mf.predict_future_days(model, daily_feat, n_future=14)
        fc_chart = future.set_index("date")["predicted_daily_sales"]
        st.line_chart(fc_chart)
        st.caption("14-day ahead daily sales extrapolation (Linear Regression on day index).")
    else:
        st.info("Train the model first: run `python run_pipeline.py` to create the forecast model.")

    st.divider()
    st.markdown(
        "**How to refresh data:** edit or replace `data/retail_sales_raw.csv`, "
        "then run `python run_pipeline.py` again."
    )


if __name__ == "__main__":
    main()
