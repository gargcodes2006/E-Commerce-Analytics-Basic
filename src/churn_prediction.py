"""
Churn Prediction Module for E-Commerce Customer Analytics.

This module creates a transparent rule-based proxy for customer churn based
on purchase inactivity, engineers behavioral features, trains a regularized
Logistic Regression classifier following strict ML best practices (train/test
split before scaling), evaluates model metrics, and persists the pipeline.
"""

from typing import Dict, Any, Tuple
import os
import sys
import warnings

# Avoid OpenMP thread memory leak warning on Windows and silence warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def prepare_churn_data(
    rfm_df: pd.DataFrame,
    customer_demographics: pd.DataFrame = None,
    inactivity_threshold_days: int = 90,
) -> pd.DataFrame:
    """Prepare customer-level dataset with target churn label and behavioral features.

    Note on Data Science Best Practices & Leakage Prevention:
    Churn is labeled based on inactivity (Recency > inactivity_threshold_days).
    To prevent direct target leakage, we do NOT use Recency directly as a feature.
    Instead, we predict churn risk from purchasing habits: Frequency, Monetary value,
    Average Order Value, and customer demographics (if available).

    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM DataFrame containing 'CustomerID', 'Recency', 'Frequency', 'Monetary'.
    customer_demographics : pd.DataFrame, optional
        Demographic information like CustomerAge.
    inactivity_threshold_days : int
        Days of inactivity used to define customer churn for this demonstration.

    Returns
    -------
    pd.DataFrame
        Dataset with behavioral features and binary 'IsChurned' target.
    """
    df = rfm_df.copy()

    # Define transparent rule-based churn proxy
    df["IsChurned"] = (df["Recency"] > inactivity_threshold_days).astype(int)

    # Feature Engineering
    df["AverageOrderValue"] = (df["Monetary"] / df["Frequency"]).round(2)

    # Merge demographics if provided
    if customer_demographics is not None and not customer_demographics.empty:
        demo_clean = customer_demographics[["CustomerID", "CustomerAge"]].drop_duplicates()
        df = df.merge(demo_clean, on="CustomerID", how="left")
        df["CustomerAge"] = df["CustomerAge"].fillna(df["CustomerAge"].median()).astype(int)
    else:
        # Default neutral age if missing
        df["CustomerAge"] = 35

    return df


def train_churn_model(
    churn_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    model_save_path: str = "models/churn_model.pkl",
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Train a Logistic Regression pipeline for customer churn prediction.

    Follows ML best practices:
    - Splits data into Train and Test sets BEFORE fitting any transformation
    - Scales features strictly within the pipeline fitted on X_train only
    - Computes unbiased test metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Persists the pipeline to disk

    Parameters
    ----------
    churn_df : pd.DataFrame
        Dataset produced by `prepare_churn_data`.
    test_size : float
        Proportion of dataset to reserve for model testing.
    random_state : int
        Seed for reproducibility.
    model_save_path : str
        Filepath where the trained pipeline will be saved.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]
        Fitted Pipeline, evaluation metrics dictionary, and test set data for plotting.
    """
    feature_cols = ["Frequency", "Monetary", "AverageOrderValue", "CustomerAge"]
    X = churn_df[feature_cols].copy()
    y = churn_df["IsChurned"].copy()

    # Train / Test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: StandardScaler + LogisticRegression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced")),
    ])

    # Fit pipeline strictly on training data
    pipeline.fit(X_train, y_train)

    # Predict on unseen test data
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_names": feature_cols,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "churn_rate_overall": round(float(y.mean()), 4),
    }

    test_data = {
        "y_true": y_test.values.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "confusion_matrix": cm,
    }

    # Save pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)

    return pipeline, metrics, test_data


def load_churn_model(model_path: str = "models/churn_model.pkl") -> Pipeline:
    """Load persisted churn model pipeline from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved joblib model.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")
    return joblib.load(model_path)


def predict_customer_churn(
    customer_features: pd.DataFrame, model_path: str = "models/churn_model.pkl"
) -> pd.DataFrame:
    """Generate churn predictions and risk scores for new or existing customers.

    Parameters
    ----------
    customer_features : pd.DataFrame
        DataFrame with columns ['Frequency', 'Monetary', 'AverageOrderValue', 'CustomerAge'].
    model_path : str
        Path to serialized model.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with 'PredictedChurn' and 'ChurnProbability'.
    """
    pipeline = load_churn_model(model_path)
    feature_cols = ["Frequency", "Monetary", "AverageOrderValue", "CustomerAge"]
    X = customer_features[feature_cols].copy()

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    result = customer_features.copy()
    result["PredictedChurn"] = preds
    result["ChurnProbability"] = probs.round(4)
    result["RiskLevel"] = pd.cut(
        result["ChurnProbability"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return result


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_data
    from src.rfm_analysis import calculate_rfm

    clean_df, _ = load_and_clean_data()
    rfm = calculate_rfm(clean_df)
    churn_df = prepare_churn_data(rfm, customer_demographics=clean_df)

    print("=== Churn Dataset Preview ===")
    print(churn_df[["CustomerID", "Recency", "Frequency", "Monetary", "CustomerAge", "IsChurned"]].head(5))

    pipeline, metrics, test_data = train_churn_model(churn_df)
    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-Score:  {metrics['f1_score']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
