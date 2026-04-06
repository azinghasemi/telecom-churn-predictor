"""
XGBoost classifier for customer churn prediction.

Uses gradient boosting with early stopping and class-imbalance correction.
Evaluation: Classification report + ROC-AUC + Confusion Matrix heatmap.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import load_data, clean_and_encode


def preprocess_for_xgboost(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    preproc = ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )

    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    # Convert sparse → dense if needed
    X_train_np = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
    X_test_np = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

    return X_train_np, X_test_np, y_train, y_test


def train_xgboost(X_train_np, X_test_np, y_train, y_test):
    dtrain = xgb.DMatrix(X_train_np, label=y_train)
    dtest = xgb.DMatrix(X_test_np, label=y_test)

    # Class imbalance weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 4,
        "eta": 0.1,       # learning rate
        "seed": 42,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dtest, "val")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    return model, y_pred, y_pred_prob


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Yes Churn"],
        yticklabels=["No Churn", "Yes Churn"],
    )
    plt.title("Confusion Matrix — XGBoost Churn Prediction")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix.png", dpi=150)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives  (TN): {tn} — correctly predicted 'No Churn'")
    print(f"False Positives (FP): {fp} — predicted 'Churn' but actually stayed")
    print(f"False Negatives (FN): {fn} — predicted 'No Churn' but actually churned")
    print(f"True Positives  (TP): {tp} — correctly predicted 'Churn'")


if __name__ == "__main__":
    df_raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_and_encode(df_raw)

    X_train_np, X_test_np, y_train, y_test = preprocess_for_xgboost(df)
    model, y_pred, y_pred_prob = train_xgboost(X_train_np, X_test_np, y_train, y_test)
    plot_confusion_matrix(y_test, y_pred)
