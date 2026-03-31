import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_data(cleaned_data_path: str):
    """Load the cleaned dataset for summary and charts."""
    if os.path.exists(cleaned_data_path):
        df = pd.read_csv(cleaned_data_path)
    else:
        df = pd.DataFrame()
    return df


MODEL_PATH = os.path.join("models", "student_performance_model.pkl")
DATA_PATH = os.path.join("data", "student_performance_cleaned.csv")

model = load_model(MODEL_PATH)
df = load_data(DATA_PATH)


st.set_page_config(
    page_title="AI-Based Student Performance Prediction",
    layout="wide",
)

st.title("🎓 AI-Based Student Performance Predictions and Analysis")
st.write(
    "This app uses a trained machine learning model to predict a student's **final score** "
    "based on study habits and attendance, and provides simple analytics."
)

st.sidebar.header("Student Input Features")

if not df.empty:
    study_min, study_max = float(df["study_hours"].min()), float(df["study_hours"].max())
    attend_min, attend_max = float(df["attendance"].min()), float(df["attendance"].max())
    prev_min, prev_max = float(df["previous_scores"].min()), float(df["previous_scores"].max())
    sleep_min, sleep_max = float(df["sleep_hours"].min()), float(df["sleep_hours"].max())
else:
    study_min, study_max = 0.0, 10.0
    attend_min, attend_max = 50.0, 100.0
    prev_min, prev_max = 40.0, 95.0
    sleep_min, sleep_max = 4.0, 9.0

study_hours = st.sidebar.slider(
    "Average Study Hours per Day",
    min_value=round(study_min, 1),
    max_value=round(study_max, 1),
    value=round((study_min + study_max) / 2, 1),
    step=0.1,
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=int(attend_min),
    max_value=int(attend_max),
    value=int((attend_min + attend_max) / 2),
    step=1,
)

previous_scores = st.sidebar.slider(
    "Previous Average Score",
    min_value=int(prev_min),
    max_value=int(prev_max),
    value=int((prev_min + prev_max) / 2),
    step=1,
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours per Night",
    min_value=round(sleep_min, 1),
    max_value=round(sleep_max, 1),
    value=round((sleep_min + sleep_max) / 2, 1),
    step=0.1,
)

input_data = pd.DataFrame(
    {
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_scores": [previous_scores],
        "sleep_hours": [sleep_hours],
    }
)

st.subheader("📌 Predicted Final Score")

if st.button("Predict Final Score"):
    predicted_score = model.predict(input_data)[0]
    predicted_score = float(np.clip(predicted_score, 0, 100))

    st.metric(
        label="Predicted Final Score (0–100)",
        value=f"{predicted_score:.1f}",
    )

    if predicted_score >= 80:
        level = "Excellent"
    elif predicted_score >= 60:
        level = "Good"
    elif predicted_score >= 40:
        level = "Needs Improvement"
    else:
        level = "At Risk"

    st.write(f"**Performance Level:** {level}")
else:
    st.info("Adjust the sliders on the left and click **Predict Final Score** to see the model output.")

st.subheader("📊 Dataset Insights")

if df.empty:
    st.warning("Cleaned dataset not found. Run `student_performance_analysis.py` first to generate it.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Summary Statistics**")
        st.dataframe(df.describe())

    with col2:
        st.markdown("**Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Study Hours vs Final Score**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x="study_hours", y="final_score", alpha=0.7, ax=ax1)
        ax1.set_xlabel("Study Hours (per day)")
        ax1.set_ylabel("Final Score")
        st.pyplot(fig1)

    with col4:
        st.markdown("**Attendance vs Final Score**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=df,
            x="attendance",
            y="final_score",
            alpha=0.7,
            color="green",
            ax=ax2,
        )
        ax2.set_xlabel("Attendance (%)")
        ax2.set_ylabel("Final Score")
        st.pyplot(fig2)

    st.markdown("**Key Insights**")
    st.write("- Students who study more hours tend to score higher in the final exam.")
    st.write("- Higher attendance generally correlates with better performance.")
    st.write("- Previous scores and sleep hours also contribute positively, but to a smaller extent.")
    st.write("- Focusing on both consistent study and high attendance can yield the largest gains.")

