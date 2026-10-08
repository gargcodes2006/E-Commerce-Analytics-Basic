"""
RFM (Recency, Frequency, Monetary) Analysis Module.

This module calculates customer-level Recency, Frequency, and Monetary metrics,
computes quintile-based RFM scores, and classifies customers into meaningful
behavioral segments (Champions, Loyal Customers, Potential Loyalists,
New Customers, At Risk, and Lost Customers).
"""

from typing import Dict, Any, Tuple
import os
import sys
from datetime import timedelta
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Descriptive explanations for business stakeholders and students
RFM_METRIC_DESCRIPTIONS = {
    "Recency": "How recently the customer made a purchase (in days). Lower values indicate recent activity.",
    "Frequency": "How often the customer purchased. Higher values indicate loyal/repeat shoppers.",
    "Monetary": "How much total money the customer spent. Higher values indicate higher financial value.",
}

SEGMENT_DESCRIPTIONS = {
    "Champions": "Top shoppers with high recency, high frequency, and high monetary spend. Advocate for brand.",
    "Loyal Customers": "Regular repeat buyers with consistent spending. Highly responsive to promotions.",
    "Potential Loyalists": "Recent shoppers with moderate spend and frequency. Great targets for upsell programs.",
    "New Customers": "Recently acquired customers with single or few purchases. Prime targets for onboarding nurture.",
    "At Risk": "High past spending or repeat orders who haven't purchased recently. Require win-back campaigns.",
    "Lost Customers": "Lowest recency, frequency, and monetary scores. Inactive or churned.",
}


def calculate_rfm(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Calculate Recency, Frequency, Monetary values and RFM scores for each customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
    reference_date : datetime, optional
        Baseline date to calculate recency (defaults to max OrderDate + 1 day).

    Returns
    -------
    pd.DataFrame
        Customer-level RFM dataframe with scores and segment assignments.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]
        )

    # Reference date is one day after latest order to avoid Recency = 0
    if reference_date is None:
        ref_date = df["OrderDate"].max() + timedelta(days=1)
    else:
        ref_date = pd.to_datetime(reference_date)

    # 1. Customer level aggregation
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("OrderDate", lambda dates: (ref_date - dates.max()).days),
            Frequency=("OrderID", "nunique"),
            Monetary=("TotalAmount", "sum")
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)

    # 2. Quintile Scoring (1 to 5)
    # Use rank(method='first') to handle ties cleanly and avoid bin edge collisions
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # 3. Rule-based segment assignment
    def segment_customer(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "New Customers"
        if r >= 3 and (f >= 2 or m >= 2):
            return "Potential Loyalists"
        if r <= 2 and (f >= 2 or m >= 3):
            return "At Risk"
        return "Lost Customers"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    return rfm


def get_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics for each customer segment.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        RFM DataFrame with 'Segment' column.

    Returns
    -------
    pd.DataFrame
        Segment summary dataframe with customer count, total revenue, averages, and shares.
    """
    if rfm_df.empty:
        return pd.DataFrame(
            columns=["Segment", "CustomerCount", "CustomerSharePct", "TotalRevenue", "RevenueSharePct", "AvgRecency", "AvgFrequency", "AvgMonetary"]
        )

    total_customers = len(rfm_df)
    total_rev = rfm_df["Monetary"].sum()

    summary = (
        rfm_df.groupby("Segment")
        .agg(
            CustomerCount=("CustomerID", "count"),
            TotalRevenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
    )

    summary["CustomerSharePct"] = (summary["CustomerCount"] / total_customers * 100).round(1)
    summary["RevenueSharePct"] = ((summary["TotalRevenue"] / total_rev * 100).round(1) if total_rev > 0 else 0.0)
    summary["TotalRevenue"] = summary["TotalRevenue"].round(2)
    summary["AvgRecency"] = summary["AvgRecency"].round(1)
    summary["AvgFrequency"] = summary["AvgFrequency"].round(1)
    summary["AvgMonetary"] = summary["AvgMonetary"].round(2)

    # Order segments logically by priority
    segment_order = [
        "Champions", "Loyal Customers", "Potential Loyalists",
        "New Customers", "At Risk", "Lost Customers"
    ]
    summary["sort_key"] = summary["Segment"].apply(
        lambda s: segment_order.index(s) if s in segment_order else 99
    )
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    return summary


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_data

    clean_df, _ = load_and_clean_data()
    rfm_df = calculate_rfm(clean_df)
    print("=== RFM DataFrame Preview ===")
    print(rfm_df.head(5))

    seg_summary = get_segment_summary(rfm_df)
    print("\n=== Customer Segment Summary ===")
    print(seg_summary)
