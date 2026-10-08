"""
Customer Segmentation Module using K-Means Clustering and PCA.

This module prepares RFM features, computes the Elbow curve and Silhouette
scores, trains an unsupervised K-Means model, projects clusters into 2D
via Principal Component Analysis (PCA), and profiles each cluster in plain language.
"""

from typing import Dict, Any, Tuple, List
import os
import sys
import warnings

# Avoid OpenMP thread memory leak warning on Windows and silence warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_elbow_and_silhouette(
    rfm_df: pd.DataFrame, min_k: int = 2, max_k: int = 8, random_state: int = 42
) -> Dict[str, List]:
    """Calculate Within-Cluster Sum of Squares (WCSS / Inertia) and Silhouette

    Scores for a range of k values to determine the optimal number of clusters.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM DataFrame containing 'Recency', 'Frequency', 'Monetary'.
    min_k : int
        Minimum number of clusters to test.
    max_k : int
        Maximum number of clusters to test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, List]
        Dictionary with 'k_values', 'inertia_values', and 'silhouette_scores'.
    """
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    # Log-transform to handle right-skewed monetary and frequency metrics, then standardize
    features_log = np.log1p(features)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_log)

    k_values = list(range(min_k, max_k + 1))
    inertia_values = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        inertia_values.append(float(kmeans.inertia_))
        sil_score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(round(float(sil_score), 4))

    return {
        "k_values": k_values,
        "inertia_values": inertia_values,
        "silhouette_scores": silhouette_scores,
    }


def perform_kmeans_clustering(
    rfm_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, PCA]:
    """Train K-Means clustering on standardized RFM features and compute 2D PCA

    projection coordinates.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM DataFrame.
    n_clusters : int
        Number of clusters (default: 4).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, KMeans, StandardScaler, PCA]
        Clustered DataFrame, fitted KMeans estimator, fitted StandardScaler, and fitted PCA.
    """
    df_clustered = rfm_df.copy()
    features = df_clustered[["Recency", "Frequency", "Monetary"]].copy()

    # 1. Transform and scale features
    features_log = np.log1p(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_log)

    # 2. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    df_clustered["Cluster"] = cluster_labels

    # 3. PCA 2D Projection for clear visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(scaled_features)
    df_clustered["PCA1"] = pca_coords[:, 0].round(3)
    df_clustered["PCA2"] = pca_coords[:, 1].round(3)

    return df_clustered, kmeans, scaler, pca


def get_cluster_profiles(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistical profiles and business descriptions for each cluster.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        DataFrame with 'Cluster', 'Recency', 'Frequency', 'Monetary'.

    Returns
    -------
    pd.DataFrame
        Descriptive summary table for each cluster.
    """
    total_customers = len(clustered_df)
    total_rev = clustered_df["Monetary"].sum()

    profiles = (
        clustered_df.groupby("Cluster")
        .agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
    )

    profiles["CustomerSharePct"] = (profiles["CustomerCount"] / total_customers * 100).round(1)
    profiles["RevenueSharePct"] = ((profiles["TotalRevenue"] / total_rev * 100).round(1) if total_rev > 0 else 0.0)
    profiles["TotalRevenue"] = profiles["TotalRevenue"].round(2)
    profiles["AvgRecency"] = profiles["AvgRecency"].round(1)
    profiles["AvgFrequency"] = profiles["AvgFrequency"].round(1)
    profiles["AvgMonetary"] = profiles["AvgMonetary"].round(2)

    # Assign plain-English label based on averages
    def describe_cluster(row):
        avg_r = row["AvgRecency"]
        avg_f = row["AvgFrequency"]
        avg_m = row["AvgMonetary"]

        if avg_f >= 5 and avg_m >= 1500 and avg_r < 60:
            return "VIP High-Spenders (Frequent, high spend, recently active)"
        elif avg_r >= 150:
            return "Lapsed / Churned Customers (Inactive for long period)"
        elif avg_r < 45 and avg_f <= 2:
            return "Recent New Shoppers (Low frequency, recently acquired)"
        elif avg_f >= 3 and avg_r < 90:
            return "Steady Regulars (Moderate frequency and spend)"
        else:
            return "Occasional Value Buyers (Infrequent purchases, moderate spend)"

    profiles["ClusterPersona"] = profiles.apply(describe_cluster, axis=1)

    return profiles


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_data
    from src.rfm_analysis import calculate_rfm

    clean_df, _ = load_and_clean_data()
    rfm = calculate_rfm(clean_df)

    elbow_results = compute_elbow_and_silhouette(rfm, min_k=2, max_k=6)
    print("=== Elbow & Silhouette Analysis ===")
    for k, wcss, sil in zip(
        elbow_results["k_values"],
        elbow_results["inertia_values"],
        elbow_results["silhouette_scores"],
    ):
        print(f"  k={k}: WCSS = {wcss:.2f}, Silhouette Score = {sil:.4f}")

    clustered_df, km, sc, pca_model = perform_kmeans_clustering(rfm, n_clusters=4)
    profiles = get_cluster_profiles(clustered_df)
    print("\n=== Cluster Profiles ===")
    print(profiles[["Cluster", "CustomerCount", "AvgRecency", "AvgFrequency", "AvgMonetary", "ClusterPersona"]])
