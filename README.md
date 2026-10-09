# KNN-Classifier
KNN classification on the California Housing dataset to classify house values into Low, Medium, and High categories.
# KNN Classifier - California Housing

## Overview
This project applies the K-Nearest Neighbors (KNN) classification algorithm to the California Housing dataset. House values are grouped into Low, Medium, and High categories for classification.

## Objectives
- Load and explore the California Housing dataset.
- Create house-value categories.
- Split data into training and testing sets.
- Standardize features and train a KNN classifier.
- Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Dataset
The California Housing dataset is loaded using scikit-learn. The target variable MedHouseVal represents median house value in units of $100,000.

Classification categories:
- Low: below $150,000
- Medium: $150,000 to below $300,000
- High: $300,000 or above

These categories are defined for this project.

## Algorithm
K-Nearest Neighbors (KNN) with k = 5. Features are standardized before training.

## Evaluation
The notebook reports accuracy, precision, recall, F1-score, and a confusion matrix. Actual results are generated when the notebook runs.

## Files
- KNN_Classifier_California_Housing.ipynb: code and evaluation
- knn_classifier_predictions.csv: actual and predicted categories
- knn_classifier_metrics.csv: accuracy metric

## How to Run
1. Open the notebook in Google Colab.
2. Run all cells.
3. Review the printed metrics and confusion matrix.

## Author
K. Sindhu






# ============================================================
# PROJECT: California Housing Price Prediction using KNN
# Algorithm: K-Nearest Neighbors (KNN) Regression
# ============================================================

# STEP 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 2: Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# STEP 3: Separate input features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)

# STEP 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# STEP 5: Create the KNN Regression model
# StandardScaler is important because KNN uses distances.
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

# STEP 6: Train the model
knn_model.fit(X_train, y_train)

print("\nKNN model trained successfully!")

# STEP 7: Predict house values
y_pred = knn_model.predict(X_test)

# STEP 8: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== KNN REGRESSION RESULTS ==========")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error:   {rmse:.4f}")
print(f"R-squared (R2 Score):      {r2:.4f}")

# Target values are measured in units of $100,000.
print(f"MAE in dollars: ${mae * 100000:,.2f}")
print(f"RMSE in dollars: ${rmse * 100000:,.2f}")

# STEP 9: Compare actual and predicted values
results = pd.DataFrame({
    "Actual House Value": y_test.to_numpy(),
    "Predicted House Value": y_pred
})

print("\nActual vs Predicted values:")
print(results.head(10).round(3))

# STEP 10: Visualize actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual House Value ($100,000 units)")
plt.ylabel("Predicted House Value ($100,000 units)")
plt.title("KNN Regression: Actual vs Predicted")
plt.plot([0, 5], [0, 5], "r--")
plt.tight_layout()
plt.show()

# STEP 11: Visualize prediction errors
errors = y_test.to_numpy() - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=40, kde=True)
plt.xlabel("Prediction Error ($100,000 units)")
plt.title("KNN Regression: Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# STEP 12: Test a sample prediction
sample = X_test.iloc[[0]]
actual_value = y_test.iloc[0]
predicted_value = knn_model.predict(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")
print(f"Actual value:    ${actual_value * 100000:,.2f}")
print(f"Predicted value: ${predicted_value * 100000:,.2f}")

# STEP 13: Save dataset and results as CSV files
df.to_csv("california_housing.csv", index=False)
results.to_csv("knn_predictions.csv", index=False)

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})
metrics.to_csv("knn_metrics.csv", index=False)

print("\nFiles saved:")
print("1. california_housing.csv")
print("2. knn_predictions.csv")
print("3. knn_metrics.csv")

print("\nKNN Regression project completed successfully!")
