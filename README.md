# Customer Churn Prediction

A machine learning project that predicts telecom customer churn using XGBoost, with a focus on handling class imbalance and verifying feature relationships before trusting model output.

## Overview

This project applies binary classification to predict whether a telecom customer will churn (cancel their service), using the IBM Telco Customer Churn dataset.

## Dataset

- Rows: 7,043 customers
- Features: 19 (after removing the customer ID column)
- Target: Churn (1 = Churned, 0 = Stayed)
- Class balance: 73.5% Stayed, 26.5% Churned

## Workflow

1. Data Cleaning
   - TotalCharges was stored as text due to 11 blank entries; converted to numeric and filled with the median
   - Removed customerID, a unique identifier with no predictive value

2. Exploratory Data Analysis
   - Verified that Contract type follows the expected real-world pattern before modeling: 42.7% churn for month-to-month contracts, 11.3% for one-year, and 2.8% for two-year contracts
   - Investigated an initially suspicious correlation involving MultipleLines_No phone service; on inspection, the churn rate difference between groups was only about 1.8 percentage points across a large sample, indicating the correlation was statistically present but not practically meaningful

3. Feature Encoding
   - Binary columns mapped to 0/1
   - Contract encoded as ordinal, reflecting its natural duration-based order
   - Remaining nominal columns encoded using One-Hot Encoding
   - Boolean columns from one-hot encoding explicitly converted to integers for compatibility with XGBoost

4. Modeling
   - Trained an XGBoost classifier, using scale_pos_weight to account for class imbalance
   - Used a stratified train-test split

5. Prediction Function
   - Built a wrapper function that accepts plain-language inputs and internally handles encoding and column alignment before calling the trained model

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 0.791 |
| Test Accuracy | 0.755 |
| Recall (Churn) | 0.79 |
| Precision (Churn) | 0.53 |
| ROC-AUC | 0.840 |

Contract type was the single most influential feature (40% of total feature importance), consistent with the pattern found during EDA.

## Key Takeaways

- A correlation can be statistically present yet practically negligible; checking the actual percentage difference behind a correlation value matters more than the coefficient alone
- For XGBoost, prioritizing recall over precision for the minority class was the appropriate choice here, since failing to flag a customer likely to churn is generally costlier than offering an unnecessary retention incentive
- Cross-checking feature relationships against known domain patterns (such as contract length and churn) before modeling builds confidence in the dataset and the resulting model

## Tech Stack

- Python, Pandas, NumPy
- XGBoost, Scikit-learn (train_test_split, classification_report, roc_auc_score)
- Matplotlib, Seaborn

## Files

customer_churn_prediction.ipynb: full analysis and modeling notebook

## How to Run

1. Clone this repository
2. Open customer_churn_prediction.ipynb in Jupyter Notebook or Google Colab
3. Download the IBM Telco Customer Churn dataset from Kaggle and place the CSV file in the same directory
4. Run all cells
