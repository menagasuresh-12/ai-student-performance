import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_synthetic_student_data(num_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)

    study_hours = np.random.uniform(0, 10, num_samples)
    attendance = np.random.uniform(50, 100, num_samples)
    previous_scores = np.random.uniform(40, 95, num_samples)
    sleep_hours = np.random.uniform(4, 9, num_samples)

    noise = np.random.normal(0, 5, num_samples)

    final_score = (
        0.4 * study_hours * 10
        + 0.3 * attendance
        + 0.2 * previous_scores
        + 1.5 * sleep_hours
        + noise
    )

    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_scores": previous_scores,
            "sleep_hours": sleep_hours,
            "final_score": final_score,
        }
    )

    return df


def load_or_create_dataset(data_path: str) -> pd.DataFrame:
    if os.path.exists(data_path):
        print(f"Loading existing dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found. Generating synthetic dataset...")
        df = generate_synthetic_student_data(num_samples=300)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Synthetic dataset saved to: {data_path}")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Data Preprocessing ---")
    print("\nData info before preprocessing:")
    print(df.info())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nNumeric columns:", list(numeric_cols))

    print("\nMissing values before filling:")
    print(df[numeric_cols].isnull().sum())

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print("\nMissing values after filling:")
    print(df[numeric_cols].isnull().sum())

    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def perform_eda(df: pd.DataFrame, output_dir: str = "plots"):
    print("\n--- Exploratory Data Analysis (EDA) ---")

    os.makedirs(output_dir, exist_ok=True)

    print("\nSummary statistics:")
    print(df.describe())

    plt.figure(figsize=(8, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Student Performance Features")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Correlation heatmap saved to: {heatmap_path}")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="study_hours", y="final_score", alpha=0.7)
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours (per day)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    study_plot_path = os.path.join(output_dir, "study_hours_vs_final_score.png")
    plt.savefig(study_plot_path)
    print(f"Study hours vs final score plot saved to: {study_plot_path}")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df, x="attendance", y="final_score", alpha=0.7, color="green"
    )
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    attendance_plot_path = os.path.join(output_dir, "attendance_vs_final_score.png")
    plt.savefig(attendance_plot_path)
    print(f"Attendance vs final score plot saved to: {attendance_plot_path}")
    plt.close()

    print("\nBasic EDA Insights (from correlations):")
    for col in df.columns:
        if col != "final_score":
            corr_value = corr_matrix.loc[col, "final_score"]
            print(f"Correlation between {col} and final_score: {corr_value:.2f}")


def train_model(df: pd.DataFrame):
    print("\n--- Model Training ---")

    feature_cols = ["study_hours", "attendance", "previous_scores", "sleep_hours"]
    target_col = "final_score"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    print("\n--- Model Evaluation ---")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return mae, mse, r2


def plot_predictions(y_test, y_pred, output_dir: str = "plots"):
    print("\n--- Visualizing Predictions ---")

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="purple", label="Predicted vs Actual")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Actual vs Predicted Final Scores")
    plt.legend()
    plt.tight_layout()

    pred_plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(pred_plot_path)
    print(f"Actual vs predicted plot saved to: {pred_plot_path}")
    plt.close()


def save_outputs(df_clean: pd.DataFrame, model, cleaned_data_path: str, model_path: str):
    print("\n--- Saving Outputs ---")

    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df_clean.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_data_path}")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained model saved to: {model_path}")


def generate_insights(df: pd.DataFrame):
    print("\n--- Insights ---")

    corr = df.corr()["final_score"].sort_values(ascending=False)
    print("\nCorrelation of each feature with final_score:")
    print(corr)

    print("\nBusiness-style insights:")
    print("1. Students who study more hours per day tend to achieve higher final scores.")
    print(
        "2. Higher attendance percentages are strongly linked with better final performance."
    )
    print(
        "3. Students with higher previous scores are more likely to maintain good final scores."
    )
    print(
        "4. Adequate sleep hours (not too low) have a positive but smaller impact on final scores."
    )
    print(
        "5. Improving both study habits and attendance together can significantly boost student performance."
    )


if __name__ == "__main__":
    data_file_path = os.path.join("data", "student_performance.csv")
    cleaned_data_path = os.path.join("data", "student_performance_cleaned.csv")
    model_file_path = os.path.join("models", "student_performance_model.pkl")

    df = load_or_create_dataset(data_file_path)

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset columns explanation:")
    print("- study_hours: Average study hours per day.")
    print("- attendance: Attendance percentage of the student.")
    print("- previous_scores: Average score from previous exams/tests.")
    print("- sleep_hours: Average sleep hours per night.")
    print("- final_score: Final exam score (target variable).")

    df_clean = preprocess_data(df)

    perform_eda(df_clean, output_dir="plots")

    model, X_test, y_test, y_pred = train_model(df_clean)

    mae, mse, r2 = evaluate_model(y_test, y_pred)

    plot_predictions(y_test, y_pred, output_dir="plots")

    save_outputs(df_clean, model, cleaned_data_path, model_file_path)

    generate_insights(df_clean)

    print("\nPipeline completed successfully.")

