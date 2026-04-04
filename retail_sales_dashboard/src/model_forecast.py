"""
Simple sales forecasting using Linear Regression on a time index.

Why Linear Regression here?
- Beginner-friendly, fast, interpretable baseline for "sales vs time"
- Works when the signal is roughly linear trend over the aggregated period
- For production, teams often upgrade to Prophet, SARIMA, or gradient boosting

We aggregate to DAILY total sales, then use day_index (0,1,2,...) as X.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODEL_PATH = MODELS_DIR / "sales_forecast_model.pkl"


def daily_sales_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: date, daily_sales."""
    dtmp = df.copy()
    dtmp["date"] = dtmp["date"].dt.floor("D")
    daily = (
        dtmp.groupby("date", as_index=False)["sales_amount"]
        .sum()
        .rename(columns={"sales_amount": "daily_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def prepare_forecast_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    X = day index (0..n-1) as column 'day_index'
    y = daily_sales
    """
    out = daily.copy()
    out["day_index"] = np.arange(len(out), dtype=float)
    X = out[["day_index"]].values
    y = out["daily_sales"].values
    return out, X, y


def train_evaluate_forecast_model(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[LinearRegression, pd.DataFrame, int, np.ndarray, np.ndarray]:
    """
    Time-ordered split: last test_size fraction of days is test set
    (more realistic than random shuffle for forecasting).
    """
    _ = random_state  # reserved for reproducibility if we add shuffled baselines later
    daily = daily_sales_series(df)
    daily_feat, X, y = prepare_forecast_features(daily)

    # Chronological split: last portion for testing
    n_train = int(len(daily_feat) * (1 - test_size))
    n_train = max(1, min(n_train, len(daily_feat) - 1))

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, daily_feat, n_train, y_test, y_pred


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n--- Model evaluation (test period) ---")
    print(f"MAE (Mean Absolute Error): {mae:,.2f}")
    print(f"MSE (Mean Squared Error):  {mse:,.2f}")
    print(f"R² score:                  {r2:.3f}")
    print("\nIn simple terms:")
    print("- MAE: typical dollars off per day on average.")
    print("- MSE: punishes big misses more than MAE.")
    print("- R²: closer to 1.0 means the line explains more of the ups and downs.")
    print("--------------------------------------\n")


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    daily_feat: pd.DataFrame,
    n_train: int,
    output_path: Path | None = None,
) -> None:
    """Plot test-period actual vs predicted daily sales."""
    output_path = output_path or (PLOTS_DIR / "forecast_actual_vs_predicted.png")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # x-axis: test dates
    test_dates = daily_feat["date"].iloc[n_train : n_train + len(y_test)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_dates, y_test, label="Actual", color="tab:blue", marker="o", markersize=3)
    ax.plot(test_dates, y_pred, label="Predicted", color="tab:orange", marker="s", markersize=3)
    ax.set_title("Forecast: Actual vs Predicted Daily Sales (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sales ($)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Forecast plot saved: {output_path}")


def save_model(model: LinearRegression, path: Path | None = None) -> None:
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {path}")


def predict_future_days(
    model: LinearRegression, daily_feat: pd.DataFrame, n_future: int = 14
) -> pd.DataFrame:
    """
    Extrapolate n_future days beyond the last observed day_index.
    """
    last_idx = daily_feat["day_index"].max()
    future_idx = np.arange(last_idx + 1, last_idx + 1 + n_future).reshape(-1, 1)
    preds = model.predict(future_idx)
    last_date = daily_feat["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_future, freq="D")
    return pd.DataFrame({"date": future_dates, "predicted_daily_sales": preds})
