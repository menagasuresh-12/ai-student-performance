# Retail Sales Analysis and Forecasting Dashboard

## Project overview

End-to-end **retail analytics** project: load and clean sales data, explore trends with visualizations, summarize **business insights**, train a simple **Linear Regression** model to forecast **daily sales** from a time index, evaluate with standard metrics, save artifacts, and explore results in a **Streamlit** dashboard.

## Tools used

- **Python 3.8+**
- **Pandas**, **NumPy** — data loading, cleaning, aggregations
- **Matplotlib**, **Seaborn** — charts and correlation heatmap
- **Scikit-learn** — Linear Regression, MAE / MSE / R²
- **Streamlit** — interactive dashboard
- **Pickle** — model persistence (standard library)

## Folder structure

```text
retail_sales_dashboard/
├── app.py                 # Streamlit dashboard
├── run_pipeline.py        # Run full analysis + training
├── requirements.txt
├── README.md
├── data/
│   ├── retail_sales_raw.csv       # created on first run if missing
│   └── retail_sales_cleaned.csv   # output of cleaning step
├── models/
│   └── sales_forecast_model.pkl   # saved after training
├── plots/                         # EDA + forecast charts from pipeline
└── src/
    ├── data_processing.py         # load, synthetic data, cleaning
    ├── eda.py                     # KPIs, plots, business insights
    └── model_forecast.py          # forecasting model + evaluation
```

## Steps performed

1. **Setup** — Install dependencies from `requirements.txt`.
2. **Data loading** — If `data/retail_sales_raw.csv` is missing, a **realistic synthetic** dataset is generated with columns: `date`, `product`, `category`, `region`, `sales_amount`, `quantity`, `profit`. To regenerate synthetic data from scratch, delete `data/retail_sales_raw.csv` (and optionally `retail_sales_cleaned.csv`) and run the pipeline again.
3. **Cleaning** — Parse dates, drop duplicates, handle missing values, enforce dtypes.
4. **EDA** — Revenue/profit/quantity totals; monthly trend; sales by category and region; top 5 products; line, bar, and correlation heatmap plots.
5. **Business insights** — Printed narrative insights (category/region leaders, seasonality hints, actions).
6. **ML forecasting** — Daily sales aggregated; feature = `day_index`; **chronological** train/test split; **Linear Regression** (interpretable baseline).
7. **Evaluation** — MAE, MSE, R² on the test period; actual vs predicted plot.
8. **Outputs** — `retail_sales_cleaned.csv`, `sales_forecast_model.pkl`, PNG plots under `plots/`.
9. **Dashboard** — `app.py` shows KPIs, top products, monthly trend, category/region charts, and optional 14-day forecast.

## Key insights (typical patterns)

Insights are **data-driven** when you run the pipeline; expect themes like:

- Which **category** drives the most revenue.
- Which **region** leads sales and profit.
- **Monthly** ups and downs (seasonal hints in synthetic data).
- **Top products** to prioritize for stock and promotions.
- How a **simple trend model** approximates near-term daily sales (upgrade path: seasonality models for production).

## How to run

From inside `retail_sales_dashboard`:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

- First pipeline run creates raw data (if needed), cleaned CSV, plots, and the model.
- Open the URL Streamlit prints (usually `http://localhost:8501`).

## Resume line (example)

**Retail Sales Analysis and Forecasting Dashboard** — Built a Python pipeline (Pandas, Scikit-learn) to clean retail transactions, analyze category/region performance, train a linear regression forecast on daily sales, and deployed an interactive Streamlit dashboard with KPIs and trend visualizations.
