"""
Decision Tree classifier for customer churn prediction.

Pipeline: One-Hot Encoding → DecisionTreeClassifier
Evaluation: 5-fold cross-validation (F1), held-out test set
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import load_data, clean_and_encode


def build_pipeline(cat_cols: list) -> Pipeline:
    preproc = ColumnTransformer(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [
            ("prep", preproc),
            (
                "clf",
                DecisionTreeClassifier(
                    class_weight="balanced",  # handle class imbalance
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate(df: pd.DataFrame):
    # Prepare features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    pipeline = build_pipeline(cat_cols)

    # Cross-validation on training set
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print("5-fold F1 scores:", scores)
    print(f"Mean F1: {scores.mean():.4f}")

    # Final evaluation on held-out test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return pipeline


if __name__ == "__main__":
    df_raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_and_encode(df_raw)
    train_and_evaluate(df)
