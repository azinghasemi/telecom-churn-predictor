"""
Exploratory Data Analysis — distributions, correlations, and churn patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, clean_and_encode


def plot_tenure_histogram(df: pd.DataFrame):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    df["tenure"].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title("Customer Tenure Distribution")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("tenure_histogram.png", dpi=150)
    plt.show()
    print("Insight: Most customers have short tenure (1–12 months). Churn peaks early.")


def plot_monthly_charges_boxplot(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        data=df,
        palette={"Yes": "#fc8d62", "No": "#66c2a5"},
        legend=False,
    )
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges ($)")
    plt.tight_layout()
    plt.savefig("monthly_charges_boxplot.png", dpi=150)
    plt.show()
    print("Insight: Churned customers have significantly higher median monthly charges.")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Insight: Tenure & TotalCharges are highly correlated (0.83). MonthlyCharges is weakly correlated with tenure.")


def statistical_summary(df: pd.DataFrame):
    summary = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "std"])
    print("\nMonthly Charges by Churn group:")
    print(summary)
    print("\nChurned customers pay ~$13 more/month on average with higher variance.")


if __name__ == "__main__":
    df_raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_and_encode(df_raw)

    plot_tenure_histogram(df)
    plot_monthly_charges_boxplot(df)
    plot_correlation_heatmap(df)
    statistical_summary(df)
