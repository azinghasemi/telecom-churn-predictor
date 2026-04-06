"""
Data loading, cleaning, and encoding for the Telco Customer Churn dataset.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    churn_rate = df["Churn"].value_counts(normalize=True)["Yes"] * 100
    print(f"Churn rate: {churn_rate:.1f}%")
    return df


def identify_column_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\n------ Numeric columns ------")
    for c in num_cols:
        print(c)
    print("\n------ Categorical columns ------")
    for c in cat_cols:
        print(c)
    return num_cols, cat_cols


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Convert TotalCharges to numeric (stored as string in raw data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print(f"'TotalCharges' missing values handled: {missing_tc} imputed with median.")

    # 2. Preserve target and drop identifier
    target = df["Churn"]
    df.drop(["customerID", "Churn"], axis=1, inplace=True)
    print("Dropped 'customerID' column and temporarily removed 'Churn'.")

    # 3. One-hot encode low-cardinality categorical columns (≤ 5 unique values)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    low_card_cols = [col for col in cat_cols if df[col].nunique() <= 5]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = ohe.fit_transform(df[low_card_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(low_card_cols))

    df.drop(columns=low_card_cols, inplace=True)
    df_encoded = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # 4. Add Churn back
    df_encoded["Churn"] = target.values
    print(f"One-hot encoded: {low_card_cols}")
    print(f"Final shape after encoding: {df_encoded.shape}")

    # 5. Verify no missing values remain
    missing_rate = df_encoded.isnull().sum() / len(df_encoded) * 100
    missing_rate = missing_rate[missing_rate > 0].sort_values(ascending=False)
    if missing_rate.empty:
        print("No missing values remaining.")
    else:
        print("Missing Value Rate (%):")
        print(missing_rate.round(2))

    return df_encoded


if __name__ == "__main__":
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    identify_column_types(df)
    df_encoded = clean_and_encode(df)
    print(df_encoded.head())
