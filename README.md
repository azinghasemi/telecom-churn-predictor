# Telecom Customer Churn Predictor

A machine learning pipeline to predict customer churn in a telecom company using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The project combines supervised classification and unsupervised segmentation to identify at-risk customers and guide retention strategy.

---

## Business Problem

Customer churn — when a subscriber cancels service — directly impacts revenue. Acquiring a new customer costs 5–25× more than retaining one. This project builds predictive models that flag high-risk churners before they leave, enabling proactive intervention.

**Churn Rate in dataset:** 26.5% (1,869 churned out of 7,043 customers)

---

## Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Shape:** 7,043 rows × 21 columns
- **Target:** `Churn` (Yes/No)
- **Features:** Demographics, account info, subscribed services, billing details

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in the `data/` folder.

---

## Project Structure

```
telecom-churn-predictor/
├── data/                          # Place dataset CSV here
├── notebooks/
│   └── churn_analysis.ipynb      # Full end-to-end analysis notebook
├── src/
│   ├── preprocessing.py          # Data cleaning & encoding
│   ├── eda.py                    # Exploratory data analysis & visualizations
│   └── models/
│       ├── decision_tree.py      # Decision Tree classifier
│       ├── xgboost_model.py      # XGBoost classifier
│       └── kmeans.py             # K-Means customer segmentation
├── requirements.txt
└── README.md
```

---

## Algorithms

### 1. Decision Tree
- Pipeline with One-Hot Encoding + `DecisionTreeClassifier`
- 5-fold cross-validation, balanced class weights
- Mean F1 Score: **0.61**

### 2. XGBoost
- Gradient boosting with early stopping
- Class imbalance handled via `scale_pos_weight`
- F1 Score (churn class): **0.63** | ROC-AUC: **0.84**

### 3. K-Means Clustering (k=4)
- Unsupervised segmentation on tenure, MonthlyCharges, TotalCharges
- Elbow method used to select optimal k
- Cluster 3 identified as highest-risk (~48% churn rate)

---

## Key Findings

| Cluster | Churn Rate | Profile |
|---------|-----------|---------|
| 0 | 5.0% | Very loyal, long-tenure, low-cost |
| 1 | 15.4% | Stable, moderate risk |
| 2 | 24.6% | Medium risk, mixed services |
| 3 | 48.2% | High risk — new, high-paying customers |

**Top churn drivers:**
- Short tenure (< 12 months)
- Month-to-month contracts
- High monthly charges

---

## Business Recommendations

1. **Cluster 3** — Launch targeted retention campaigns with long-term discount incentives
2. **Cluster 2** — Collect proactive feedback before contract renewal dates
3. **Clusters 0 & 1** — Upsell bundled services to maximize revenue from loyal customers

---

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/churn_analysis.ipynb
```

Or run individual modules:

```bash
python src/preprocessing.py
python src/eda.py
python src/models/decision_tree.py
python src/models/xgboost_model.py
python src/models/kmeans.py
```

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

See `requirements.txt` for full list.

---

## Results Summary

| Model | Accuracy | F1 (Churn) | ROC-AUC |
|-------|----------|------------|---------|
| Decision Tree | 76% | 0.62 | — |
| XGBoost | 76% | 0.63 | 0.84 |
