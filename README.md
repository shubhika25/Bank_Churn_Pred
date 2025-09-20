# Bank Churn Prediction

Predict customer churn for a retail bank using demographic and transactional features. This is a full-stack machine learning project with an interactive Streamlit frontend and deployment-ready backend.

<img width="537" height="780" alt="image" src="https://github.com/user-attachments/assets/a9519bc4-9b70-4764-b6ea-d16d320ac791" />

## 📌Overview

The goal of this project is to build a classification model that can identify customers likely to churn, enabling the bank to take proactive retention measures. The application includes:

Cleaned and preprocessed customer data

Comprehensive exploratory data analysis (EDA) with feature engineering

Multiple machine learning models evaluated and tuned

Interactive frontend using Streamlit for predictions and insights

Deployment-ready code structure

## Tech Stack

Python, pandas, NumPy

scikit-learn, XGBoost, Gradient Boosting, Random Forest

Streamlit for the user interface <ss>

Matplotlib and Seaborn for visualizations

Jupyter Notebook for EDA and model development

Git for version control

## Model Evaluation and Selection

Approach:

Train-test split and cross-validation for robust performance estimation.

Hyperparameter tuning with GridSearchCV for optimized results.

Evaluated six algorithms for accuracy, precision, recall, F1, and ROC-AUC.

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ----- | ------- |
| **XGBoost**         | 0.966    | 0.900     | 0.889  | 0.895 | 0.991   |
| Gradient Boosting   | 0.957    | 0.841     | 0.898  | 0.869 | 0.990   |
| Random Forest       | 0.953    | 0.846     | 0.865  | 0.855 | 0.984   |
| Decision Tree       | 0.923    | 0.730     | 0.825  | 0.775 | 0.883   |
| Logistic Regression | 0.849    | 0.520     | 0.775  | 0.622 | 0.913   |
| SVM                 | 0.848    | 0.517     | 0.785  | 0.623 | 0.914   |


Ranking (Based on Confusion Matrices):

1. XGBoost – Best overall balance between precision and recall.
2. Gradient Boosting – Highest recall; ideal if missing churners is costly.
3. Random Forest – Stable performance but slightly weaker on false positives/negatives.
