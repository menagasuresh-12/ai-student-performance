"""
Retail Sales Analysis and Forecasting — full end-to-end pipeline.

Run from the `retail_sales_dashboard` folder:

    python run_pipeline.py

Steps:
1. Load or generate raw data
2. Clean and save CSV
3. EDA metrics, plots, business insights
4. Train forecasting model, evaluate, plot, save model
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (folder containing this file)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import (
    CLEAN_CSV,
    RAW_CSV,
    clean_retail_data,
    explain_columns,
    load_or_create_raw_dataset,
)
from src.eda import (
    compute_kpis,
    create_eda_plots,
    monthly_sales_trend,
    print_business_insights,
    sales_by_category,
    sales_by_region,
    top_n_products,
)
from src import model_forecast as mf


def main() -> None:
    print("=" * 60)
    print("Retail Sales Analysis and Forecasting Dashboard — Pipeline")
    print("=" * 60)

    # ----- 1–2. Load & clean -----
    df_raw = load_or_create_raw_dataset(RAW_CSV)
    print("\n--- First 5 rows (raw) ---")
    print(df_raw.head())

    explain_columns()

    df_clean = clean_retail_data(df_raw)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"\nCleaned dataset saved to: {CLEAN_CSV}")

    # ----- 4. EDA summaries -----
    print("\n--- KPI totals ---")
    kpis = compute_kpis(df_clean)
    print(f"Total revenue (sales): ${kpis['total_revenue']:,.2f}")
    print(f"Total profit:          ${kpis['total_profit']:,.2f}")
    print(f"Total quantity sold:   {kpis['total_quantity']:,}")

    print("\n--- Monthly sales trend (head) ---")
    print(monthly_sales_trend(df_clean).head(10))

    print("\n--- Sales by category ---")
    print(sales_by_category(df_clean))

    print("\n--- Sales by region ---")
    print(sales_by_region(df_clean))

    print("\n--- Top 5 products by revenue ---")
    print(top_n_products(df_clean, 5))

    create_eda_plots(df_clean)

    # ----- 5. Business insights (printed) -----
    print_business_insights(df_clean)

    # ----- 6–8. Forecast model -----
    print(
        "\n--- Forecasting model (Linear Regression on daily sales vs time index) ---\n"
        "We use a simple linear trend as a teaching baseline. Real retailers often\n"
        "use seasonality models; this project keeps the math easy to explain.\n"
    )
    model, daily_feat, n_train, y_test, y_pred = mf.train_evaluate_forecast_model(
        df_clean, test_size=0.2
    )
    mf.print_metrics(y_test, y_pred)
    mf.plot_actual_vs_predicted(y_test, y_pred, daily_feat, n_train)

    future = mf.predict_future_days(model, daily_feat, n_future=14)
    print("--- Next 14 days predicted daily sales (extrapolation) ---")
    print(future.to_string(index=False))

    mf.save_model(model)

    print("\nPipeline finished successfully.")
    print(f"- Cleaned data: {CLEAN_CSV}")
    print(f"- Model:        {mf.MODEL_PATH}")
    print(f"- Plots folder: {ROOT / 'plots'}")
    print("\nLaunch dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
