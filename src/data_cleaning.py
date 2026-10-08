"""
Data Cleaning Module for E-Commerce Customer Analytics.

This module provides a robust, reusable pipeline to clean raw transaction
data by handling missing values, duplicate records, invalid amounts/quantities,
and improper data types.
"""

from typing import Dict, Tuple, Any
import os
import pandas as pd
import numpy as np


def clean_ecommerce_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw e-commerce transaction data and return the cleaned DataFrame

    along with cleaning audit statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions dataframe containing e-commerce records.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple containing:
        - Cleaned DataFrame ready for analysis and modeling.
        - Dictionary summarizing data cleaning metrics.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    required_columns = [
        "OrderID", "CustomerID", "OrderDate", "Product", "Category",
        "Quantity", "UnitPrice", "TotalAmount", "Country"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Track metrics
    original_rows = len(df)
    missing_values_before = int(df.isnull().sum().sum())

    # Step 1: Remove duplicate rows
    duplicates_count = int(df.duplicated().sum())
    cleaned_df = df.drop_duplicates().copy()

    # Step 2: Handle missing Customer IDs (cannot link to customer profiles/RFM)
    missing_customers = int(cleaned_df["CustomerID"].isnull().sum())
    cleaned_df = cleaned_df.dropna(subset=["CustomerID"]).copy()

    # Step 3: Handle UnitPrice and TotalAmount consistency
    # If UnitPrice is missing but TotalAmount & Quantity exist, impute UnitPrice
    price_impute_mask = cleaned_df["UnitPrice"].isnull() & cleaned_df["TotalAmount"].notnull() & (cleaned_df["Quantity"] > 0)
    cleaned_df.loc[price_impute_mask, "UnitPrice"] = (
        cleaned_df.loc[price_impute_mask, "TotalAmount"] / cleaned_df.loc[price_impute_mask, "Quantity"]
    )
    # Drop any remaining rows with missing UnitPrice
    cleaned_df = cleaned_df.dropna(subset=["UnitPrice"]).copy()

    # Step 4: Filter out invalid negative or zero quantities and prices (returns/cancellations)
    invalid_mask = (cleaned_df["Quantity"] <= 0) | (cleaned_df["UnitPrice"] <= 0)
    invalid_values_removed = int(invalid_mask.sum())
    cleaned_df = cleaned_df[~invalid_mask].copy()

    # Recompute/verify TotalAmount = Quantity * UnitPrice to ensure consistency
    cleaned_df["TotalAmount"] = (cleaned_df["Quantity"] * cleaned_df["UnitPrice"]).round(2)

    # Step 5: Convert and correct data types
    cleaned_df["OrderDate"] = pd.to_datetime(cleaned_df["OrderDate"], errors="coerce")
    # Drop rows where OrderDate could not be parsed
    cleaned_df = cleaned_df.dropna(subset=["OrderDate"]).copy()

    cleaned_df["CustomerID"] = cleaned_df["CustomerID"].astype(str).str.strip()
    cleaned_df["OrderID"] = cleaned_df["OrderID"].astype(str).str.strip()
    cleaned_df["Product"] = cleaned_df["Product"].astype(str).str.strip()
    cleaned_df["Category"] = cleaned_df["Category"].astype(str).str.strip()
    cleaned_df["Country"] = cleaned_df["Country"].astype(str).str.strip()
    cleaned_df["Quantity"] = cleaned_df["Quantity"].astype(int)
    cleaned_df["UnitPrice"] = cleaned_df["UnitPrice"].astype(float)
    cleaned_df["TotalAmount"] = cleaned_df["TotalAmount"].astype(float)

    # Step 6: Handle optional demographic columns if present
    if "CustomerAge" in cleaned_df.columns:
        median_age = cleaned_df["CustomerAge"].dropna().median()
        cleaned_df["CustomerAge"] = cleaned_df["CustomerAge"].fillna(median_age).astype(int)

    if "Gender" in cleaned_df.columns:
        cleaned_df["Gender"] = cleaned_df["Gender"].fillna("Other").astype(str).str.strip()

    final_rows = len(cleaned_df)

    cleaning_stats = {
        "original_rows": original_rows,
        "removed_duplicates": duplicates_count,
        "missing_customer_ids": missing_customers,
        "invalid_values_removed": invalid_values_removed,
        "missing_values_before": missing_values_before,
        "final_rows": final_rows,
    }

    return cleaned_df, cleaning_stats


def load_and_clean_data(csv_path: str = "data/ecommerce_data.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load transactions from CSV and process them through the data cleaning pipeline.

    Parameters
    ----------
    csv_path : str
        Relative or absolute path to the CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and cleaning statistics dictionary.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    return clean_ecommerce_data(raw_df)


if __name__ == "__main__":
    df_clean, stats = load_and_clean_data()
    print("=== Data Cleaning Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nCleaned Data Preview:")
    print(df_clean.head(3))
