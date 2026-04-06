"""
K-Means customer segmentation.

Uses tenure, MonthlyCharges, TotalCharges to segment customers into risk clusters.
Elbow method determines optimal k. PCA used for 2D visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import load_data, clean_and_encode


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k (K-Means)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kmeans_elbow.png", dpi=150)
    plt.show()


def fit_kmeans(X_scaled: np.ndarray, k: int = 4) -> KMeans:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    return km


def plot_clusters(X_scaled: np.ndarray, labels: np.ndarray):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Customer Segments (K-Means Clusters via PCA)")
    plt.xlabel("pca1")
    plt.ylabel("pca2")
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150)
    plt.show()


def analyze_churn_by_cluster(df: pd.DataFrame):
    churn_table = (
        df.groupby("cluster")["Churn"]
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={"No": "% Stay", "Yes": "% Churn"})
        .round(3)
    )
    print("\nChurn rate per cluster:")
    print(churn_table)

    interpretation = {
        0: "Very loyal, low-risk (safe segment)",
        1: "Stable users, low churn risk",
        2: "Moderate churn risk — consider retention efforts",
        3: "HIGH RISK — new, high-paying, or dissatisfied customers",
    }
    print("\nCluster interpretation:")
    for k, v in interpretation.items():
        pct = churn_table.loc[k, "% Churn"] * 100 if k in churn_table.index else "N/A"
        print(f"  Cluster {k} ({pct:.1f}% churn): {v}")


if __name__ == "__main__":
    df_raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_and_encode(df_raw)

    X_num = df[NUMERIC_FEATURES]
    X_scaled = StandardScaler().fit_transform(X_num)

    # Find optimal k
    elbow_method(X_scaled)

    # Fit with k=4 (elbow point)
    km = fit_kmeans(X_scaled, k=4)
    df["cluster"] = km.labels_

    # Preview
    print(df[NUMERIC_FEATURES + ["cluster"]].head())

    # Visualise
    plot_clusters(X_scaled, km.labels_)

    # Business insight
    analyze_churn_by_cluster(df)
