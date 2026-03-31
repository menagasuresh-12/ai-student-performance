## AI-Based Student Performance Prediction and Analysis

### Overview

This project builds an **end-to-end machine learning pipeline** to predict and analyze student performance using features such as **study hours, attendance, previous scores, and sleep hours**. The goal is to show how data analysis and a simple AI model can help educators understand which factors most influence student success.

### Tools and Technologies

- **Programming Language**: Python (3.8+)
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (Linear Regression)
- **Model Persistence**: Pickle (Python standard library)

### Project Structure

- `data/`
  - `student_performance.csv` – Raw or synthetic student performance dataset.
  - `student_performance_cleaned.csv` – Cleaned dataset after preprocessing.
- `models/`
  - `student_performance_model.pkl` – Trained regression model saved with pickle.
- `src/`
  - `student_performance_analysis.py` – Main script for data loading, EDA, modeling, and evaluation.
- `plots/`
  - Generated plots (correlation heatmap, feature relationships, prediction results).
- `requirements.txt`
- `README.md`

### Steps Performed

1. **Project Setup**
   - Created a structured folder layout for data, models, and source code.
   - Installed required libraries listed in `requirements.txt`.

2. **Data Loading**
   - Loaded data from `data/student_performance.csv`.
   - If the dataset is missing, a realistic **synthetic dataset** is automatically generated.
   - Reviewed the first few rows and understood each column:
     - `study_hours`, `attendance`, `previous_scores`, `sleep_hours`, `final_score`.

3. **Data Preprocessing**
   - Inspected data types and missing values.
   - Handled missing values by filling numeric columns with their mean.
   - Ensured all numerical features are of the correct type for modeling.

4. **Exploratory Data Analysis (EDA)**
   - Computed summary statistics (mean, min, max, quartiles).
   - Created and saved:
     - **Correlation heatmap** for all numeric features.
     - **Study hours vs final score** scatter plot.
     - **Attendance vs final score** scatter plot.
   - Interpreted how each feature correlates with the final score.

5. **Machine Learning Model**
   - Selected a **Linear Regression** model as a simple, interpretable baseline.
   - Split the dataset into **training (80%)** and **testing (20%)** sets.
   - Trained the model to predict `final_score` from `study_hours`, `attendance`, `previous_scores`, and `sleep_hours`.

6. **Model Evaluation**
   - Evaluated model performance using:
     - **Mean Absolute Error (MAE)**
     - **Mean Squared Error (MSE)**
     - **R² Score**
   - Printed results in a clear, readable format.

7. **Result Visualization**
   - Plotted **Actual vs Predicted Final Scores** to visually inspect model performance.
   - Saved all plots to the `plots/` directory.

8. **Outputs**
   - Saved the **cleaned dataset** to `data/student_performance_cleaned.csv`.
   - Saved the **trained model** to `models/student_performance_model.pkl` using pickle.

### Key Insights

From the analysis and model:

1. **Study hours are strongly linked to higher final scores**, suggesting that consistent, focused study time is a key driver of performance.
2. **Attendance shows a strong positive correlation with final scores**, indicating that students who attend classes regularly tend to perform better.
3. **Previous scores are a good predictor of future performance**, helping identify high-risk students early.
4. **Sleep hours have a moderate positive effect**, highlighting the importance of healthy sleep habits for academic success.
5. **Improving both study habits and attendance simultaneously** can lead to significant gains in final scores across the student population.

### How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script from the project root:

   ```bash
   python src/student_performance_analysis.py
   ```

3. Check:
   - Console output for preprocessing details, evaluation metrics, and insights.
   - `plots/` folder for generated visualizations.
   - `data/` folder for raw and cleaned datasets.
   - `models/` folder for the saved model.

### Possible Extensions

- Experiment with other models (e.g., Decision Tree, Random Forest).
- Add demographic attributes (age, gender, parental education) for deeper analysis.
- Build a simple web dashboard to input student features and show predicted scores in real time.

