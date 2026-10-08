"""
Sales and Customer Analysis Module for E-Commerce Customer Analytics.

This module computes core business KPIs, sales trends, product rankings,
geographic distribution, and customer metrics. It also includes an embedded
SQLite integration to enable SQL-based querying on clean transactional data.
"""

from typing import Dict, Any, Tuple
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level business Key Performance Indicators (KPIs).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    Dict[str, Any]
        Dictionary with Total Revenue, Total Orders, Total Customers, and Average Order Value.
    """
    total_revenue = float(df["TotalAmount"].sum()) if not df.empty else 0.0
    total_orders = int(df["OrderID"].nunique()) if not df.empty else 0
    total_customers = int(df["CustomerID"].nunique()) if not df.empty else 0
    avg_order_value = round(total_revenue / total_orders, 2) if total_orders > 0 else 0.0

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
    }


def get_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly revenue and order volume trends.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with YearMonth, Revenue, Orders, and Customers.
    """
    if df.empty:
        return pd.DataFrame(columns=["YearMonth", "Revenue", "Orders", "Customers"])

    temp = df.copy()
    temp["YearMonth"] = temp["OrderDate"].dt.to_period("M").astype(str)

    monthly = (
        temp.groupby("YearMonth")
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("OrderID", "nunique"),
            Customers=("CustomerID", "nunique")
        )
        .reset_index()
        .sort_values("YearMonth")
    )
    monthly["Revenue"] = monthly["Revenue"].round(2)
    return monthly


def get_daily_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily order volume and revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date, Orders, and Revenue.
    """
    if df.empty:
        return pd.DataFrame(columns=["Date", "Orders", "Revenue"])

    daily = (
        df.groupby(df["OrderDate"].dt.date)
        .agg(
            Orders=("OrderID", "nunique"),
            Revenue=("TotalAmount", "sum")
        )
        .reset_index()
        .rename(columns={"OrderDate": "Date"})
        .sort_values("Date")
    )
    daily["Revenue"] = daily["Revenue"].round(2)
    return daily


def get_revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregate revenue and quantity by product category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    pd.DataFrame
        Category breakdown sorted descending by revenue.
    """
    if df.empty:
        return pd.DataFrame(columns=["Category", "Revenue", "UnitsSold", "SharePercentage"])

    cat_summary = (
        df.groupby("Category")
        .agg(
            Revenue=("TotalAmount", "sum"),
            UnitsSold=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    total_rev = cat_summary["Revenue"].sum()
    cat_summary["SharePercentage"] = (
        (cat_summary["Revenue"] / total_rev * 100).round(1) if total_rev > 0 else 0.0
    )
    cat_summary["Revenue"] = cat_summary["Revenue"].round(2)
    return cat_summary


def get_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify the top N best-performing products by total revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.
    top_n : int
        Number of products to return.

    Returns
    -------
    pd.DataFrame
        Top products ranking table.
    """
    if df.empty:
        return pd.DataFrame(columns=["Product", "Category", "Revenue", "UnitsSold", "Orders"])

    prod_summary = (
        df.groupby(["Product", "Category"])
        .agg(
            Revenue=("TotalAmount", "sum"),
            UnitsSold=("Quantity", "sum"),
            Orders=("OrderID", "nunique")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )
    prod_summary["Revenue"] = prod_summary["Revenue"].round(2)
    return prod_summary


def get_revenue_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and unique buyers across countries.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    pd.DataFrame
        Country performance summary.
    """
    if df.empty:
        return pd.DataFrame(columns=["Country", "Revenue", "Orders", "Customers"])

    country_summary = (
        df.groupby("Country")
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("OrderID", "nunique"),
            Customers=("CustomerID", "nunique")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    country_summary["Revenue"] = country_summary["Revenue"].round(2)
    return country_summary


def get_customer_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate customer-level summary table with total orders, spending, AOV, and last purchase date.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    pd.DataFrame
        Customer summary table.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["CustomerID", "TotalOrders", "TotalSpending", "AverageOrderValue", "LastPurchaseDate"]
        )

    cust_summary = (
        df.groupby("CustomerID")
        .agg(
            TotalOrders=("OrderID", "nunique"),
            TotalSpending=("TotalAmount", "sum"),
            LastPurchaseDate=("OrderDate", "max")
        )
        .reset_index()
    )
    cust_summary["AverageOrderValue"] = (
        cust_summary["TotalSpending"] / cust_summary["TotalOrders"]
    ).round(2)
    cust_summary["TotalSpending"] = cust_summary["TotalSpending"].round(2)
    cust_summary["LastPurchaseDate"] = cust_summary["LastPurchaseDate"].dt.strftime("%Y-%m-%d")

    return cust_summary.sort_values("TotalSpending", ascending=False)


def get_new_vs_returning_customers(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze single-order vs repeat-order customer metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.

    Returns
    -------
    Dict[str, Any]
        Breakdown of new (1 order) vs returning (>1 order) customers and revenue.
    """
    if df.empty:
        return {
            "single_order_customers": 0,
            "repeat_customers": 0,
            "repeat_rate_pct": 0.0,
            "single_order_revenue": 0.0,
            "repeat_order_revenue": 0.0,
        }

    orders_per_cust = df.groupby("CustomerID").agg(
        OrderCount=("OrderID", "nunique"),
        TotalSpend=("TotalAmount", "sum")
    )

    single_buyers = orders_per_cust[orders_per_cust["OrderCount"] == 1]
    repeat_buyers = orders_per_cust[orders_per_cust["OrderCount"] > 1]

    total_customers = len(orders_per_cust)
    repeat_count = len(repeat_buyers)
    repeat_rate = round((repeat_count / total_customers * 100), 2) if total_customers > 0 else 0.0

    return {
        "single_order_customers": len(single_buyers),
        "repeat_customers": repeat_count,
        "repeat_rate_pct": repeat_rate,
        "single_order_revenue": round(float(single_buyers["TotalSpend"].sum()), 2),
        "repeat_order_revenue": round(float(repeat_buyers["TotalSpend"].sum()), 2),
    }


def generate_business_insights(df: pd.DataFrame, rfm_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Generate dynamic data-driven business insights and strategic recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
    rfm_df : pd.DataFrame, optional
        Calculated RFM dataframe with segment assignments.

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated business metrics and actionable recommendations.
    """
    if df.empty:
        return {
            "insights": {},
            "recommendations": []
        }

    # 1. Category Insights
    cat_summary = get_revenue_by_category(df)
    top_cat = cat_summary.iloc[0] if not cat_summary.empty else None

    # 2. Product Insights
    top_prod_summary = get_top_products(df, top_n=1)
    top_prod = top_prod_summary.iloc[0] if not top_prod_summary.empty else None

    # 3. Monthly Trends
    monthly = get_monthly_revenue(df)
    if len(monthly) >= 2:
        last_month_rev = monthly.iloc[-1]["Revenue"]
        prev_month_rev = monthly.iloc[-2]["Revenue"]
        mom_growth = round(((last_month_rev - prev_month_rev) / prev_month_rev * 100), 1) if prev_month_rev > 0 else 0.0
    else:
        mom_growth = 0.0

    # 4. Customer concentration (Pareto principle: top 10% customers)
    cust_totals = df.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False)
    top_10_pct_count = max(1, int(len(cust_totals) * 0.10))
    top_10_pct_rev = cust_totals.head(top_10_pct_count).sum()
    total_rev = df["TotalAmount"].sum()
    top_10_pct_share = round((top_10_pct_rev / total_rev * 100), 1) if total_rev > 0 else 0.0

    # 5. Segment breakdown if RFM provided
    top_segment_name = "N/A"
    top_segment_rev_share = 0.0
    at_risk_count = 0
    at_risk_pct = 0.0

    if rfm_df is not None and not rfm_df.empty:
        from src.rfm_analysis import get_segment_summary
        seg_summary = get_segment_summary(rfm_df)
        if not seg_summary.empty:
            top_seg_row = seg_summary.sort_values("TotalRevenue", ascending=False).iloc[0]
            top_segment_name = top_seg_row["Segment"]
            top_segment_rev_share = top_seg_row["RevenueSharePct"]

            at_risk_rows = seg_summary[seg_summary["Segment"].isin(["At Risk", "Lost Customers"])]
            at_risk_count = int(at_risk_rows["CustomerCount"].sum())
            at_risk_pct = round(at_risk_count / len(rfm_df) * 100, 1)

    insights = {
        "top_category": str(top_cat["Category"]) if top_cat is not None else "N/A",
        "top_category_revenue": float(round(top_cat["Revenue"], 2)) if top_cat is not None else 0.0,
        "top_category_share": float(top_cat["SharePercentage"]) if top_cat is not None else 0.0,
        "best_product": str(top_prod["Product"]) if top_prod is not None else "N/A",
        "best_product_revenue": float(round(top_prod["Revenue"], 2)) if top_prod is not None else 0.0,
        "best_product_units": int(top_prod["UnitsSold"]) if top_prod is not None else 0,
        "top_segment": str(top_segment_name),
        "top_segment_share": float(top_segment_rev_share),
        "at_risk_customers": int(at_risk_count),
        "at_risk_pct": float(at_risk_pct),
        "top_10_pct_customer_share": float(top_10_pct_share),
        "mom_growth_pct": float(mom_growth),
    }

    # Data-backed actionable recommendations
    recommendations = [
        f"**Target Retention Offers for Inactive Customers**: {at_risk_count} customers ({at_risk_pct}% of customer base) are in the 'At Risk' or 'Lost' segments. Deploy automated email win-back offers with a limited-time incentive.",
        f"**Protect High-Value Concentration**: The top 10% of customers account for {top_10_pct_share}% of total store revenue. Implement a dedicated VIP loyalty tier with personalized support and exclusive previews.",
        f"**Promote Category Leader**: '{insights['top_category']}' dominates total store sales with {insights['top_category_share']}% revenue share. Bundle top items like '{insights['best_product']}' with higher-margin accessories to lift average order value.",
        "**Nurture Single-Purchase Buyers**: Convert one-time shoppers by triggering an onboarding discount sequence 14 days after their initial purchase.",
    ]

    return {
        "insights": insights,
        "recommendations": recommendations,
    }


def init_sqlite_db(df: pd.DataFrame, db_path: str = "data/ecommerce.db") -> None:
    """Populate or update SQLite database table from cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe.
    db_path : str
        Path to SQLite database file.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    temp_df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(temp_df["OrderDate"]):
        temp_df["OrderDate"] = temp_df["OrderDate"].dt.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        temp_df.to_sql("ecommerce_orders", conn, if_exists="replace", index=False)


def execute_sql_query(query: str, db_path: str = "data/ecommerce.db") -> pd.DataFrame:
    """Execute a read-only SQL query on the SQLite database and return results as a DataFrame.

    Parameters
    ----------
    query : str
        SQL query string.
    db_path : str
        Path to SQLite database file.

    Returns
    -------
    pd.DataFrame
        Query execution results.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please initialize the database first.")

    with sqlite3.connect(db_path) as conn:
        result_df = pd.read_sql_query(query, conn)
    return result_df


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_data

    clean_df, _ = load_and_clean_data()
    kpis = compute_kpis(clean_df)
    print("=== Core Business KPIs ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    init_sqlite_db(clean_df)
    print("\nInitialized SQLite database at data/ecommerce.db")

    sample_sql = "SELECT Category, ROUND(SUM(TotalAmount), 2) AS Revenue FROM ecommerce_orders GROUP BY Category ORDER BY Revenue DESC"
    sql_res = execute_sql_query(sample_sql)
    print("\n=== SQL Test Query Result ===")
    print(sql_res)
