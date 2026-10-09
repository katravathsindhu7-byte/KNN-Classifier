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
