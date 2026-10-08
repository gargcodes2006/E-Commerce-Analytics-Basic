"""
E-Commerce Customer Analytics Dashboard
A comprehensive portfolio Data Science web application built with Streamlit.

Author: Data Science Internship Portfolio
Technologies: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQLite, Streamlit
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure page metadata and layout
st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set plotting styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_cleaning import load_and_clean_data, clean_ecommerce_data
from src.analysis import (
    compute_kpis,
    get_monthly_revenue,
    get_daily_orders,
    get_revenue_by_category,
    get_top_products,
    get_revenue_by_country,
    get_customer_summary_table,
    get_new_vs_returning_customers,
    generate_business_insights,
    init_sqlite_db,
    execute_sql_query,
)
from src.rfm_analysis import calculate_rfm, get_segment_summary, SEGMENT_DESCRIPTIONS, RFM_METRIC_DESCRIPTIONS
from src.customer_segmentation import (
    compute_elbow_and_silhouette,
    perform_kmeans_clustering,
    get_cluster_profiles,
)
from src.churn_prediction import (
    prepare_churn_data,
    train_churn_model,
    load_churn_model,
    predict_customer_churn,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ecommerce_data.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ecommerce.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")


@st.cache_data(show_spinner=False)
def load_data():
    """Load and clean dataset with caching for fast performance."""
    if not os.path.exists(DATA_PATH):
        return None, None
    try:
        clean_df, cleaning_stats = load_and_clean_data(DATA_PATH)
        # Ensure SQLite database is initialized
        init_sqlite_db(clean_df, DB_PATH)
        return clean_df, cleaning_stats
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
raw_df, cleaning_stats = load_data()

# Handle missing or invalid dataset gracefully
if raw_df is None or raw_df.empty:
    st.error("⚠️ Dataset not found or could not be loaded!")
    st.info(
        "Please ensure `data/ecommerce_data.csv` exists. You can generate a sample dataset by running:\n\n"
        "```bash\npython src/generate_data.py\n```"
    )
    if st.button("Generate Sample Dataset Now"):
        from src.generate_data import generate_ecommerce_data
        with st.spinner("Generating sample dataset..."):
            generate_ecommerce_data(output_path=DATA_PATH)
            st.cache_data.clear()
            st.rerun()
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
st.sidebar.title("🛍️ E-Commerce Analytics")
st.sidebar.caption("Portfolio & Internship Demonstration")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Sales Analysis",
        "Customer Analysis",
        "RFM Segmentation",
        "K-Means Clustering",
        "Churn Prediction",
        "Business Insights",
        "SQL Explorer",
        "Data Preview & Cleaning Stats",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Global Filters")

# Date Filter
min_date = raw_df["OrderDate"].min().date()
max_date = raw_df["OrderDate"].max().date()
date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Category Filter
all_categories = sorted(raw_df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=all_categories,
    default=all_categories,
)

# Country Filter
all_countries = sorted(raw_df["Country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=all_countries,
    default=all_countries,
)

# Gender Filter
all_genders = sorted(raw_df["Gender"].dropna().unique().tolist())
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=all_genders,
    default=all_genders,
)

# Reset filters button
if st.sidebar.button("Reset Filters"):
    st.rerun()

# Apply Filters
filtered_df = raw_df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["OrderDate"].dt.date >= start_d) & (filtered_df["OrderDate"].dt.date <= end_d)
    ]
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_d = date_range[0]
    filtered_df = filtered_df[filtered_df["OrderDate"].dt.date >= start_d]

if selected_categories:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_genders:
    filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
else:
    filtered_df = filtered_df.iloc[0:0]

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Filtered Records**: {len(filtered_df):,} of {len(raw_df):,} rows\n\n"
    f"**Active Customers**: {filtered_df['CustomerID'].nunique():,}"
)

# Check if filtered dataframe is empty
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filter criteria. Please broaden your sidebar filters.")
    st.stop()


# ---------------------------------------------------------
# Helper function for rendering KPI metrics cards
# ---------------------------------------------------------
def render_kpis(df):
    kpis = compute_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Revenue",
            value=f"${kpis['total_revenue']:,.2f}",
            help="Total gross sales revenue from completed orders.",
        )
    with c2:
        st.metric(
            label="Total Orders",
            value=f"{kpis['total_orders']:,}",
            help="Total number of unique customer purchase transactions.",
        )
    with c3:
        st.metric(
            label="Total Customers",
            value=f"{kpis['total_customers']:,}",
            help="Number of unique customer accounts with at least one order.",
        )
    with c4:
        st.metric(
            label="Average Order Value (AOV)",
            value=f"${kpis['avg_order_value']:,.2f}",
            help="Average dollar amount spent per order transaction.",
        )


# =========================================================
# SECTION 1: DASHBOARD OVERVIEW
# =========================================================
if nav_selection == "Dashboard Overview":
    st.title("E-Commerce Customer Analytics Dashboard")
    st.markdown(
        "Welcome to the **E-Commerce Customer Analytics** platform. "
        "This interactive dashboard enables data-driven exploration of sales dynamics, "
        "customer retention, RFM behavioral segmentation, and machine-learning-based churn risk."
    )
    st.markdown("---")

    render_kpis(filtered_df)
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Monthly Revenue Trend")
        monthly_df = get_monthly_revenue(filtered_df)
        if not monthly_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(monthly_df["YearMonth"], monthly_df["Revenue"], marker="o", color="#1E88E5", linewidth=2.5)
            ax.set_title("Revenue Growth Over Time ($)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Year-Month", fontsize=10)
            ax.set_ylabel("Revenue ($)", fontsize=10)
            ax.tick_params(axis="x", rotation=45)
            # Annotate peaks
            for i, txt in enumerate(monthly_df["Revenue"]):
                if i % 2 == 0 or i == len(monthly_df) - 1:
                    ax.annotate(f"${int(txt):,}", (monthly_df["YearMonth"].iloc[i], txt), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No monthly data available for current selection.")

    with col_right:
        st.subheader("🏷️ Revenue by Category")
        cat_df = get_revenue_by_category(filtered_df)
        if not cat_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = sns.color_palette("Blues_r", len(cat_df))
            bars = ax.bar(cat_df["Category"], cat_df["Revenue"], color=colors)
            ax.set_title("Revenue by Product Category ($)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Category", fontsize=10)
            ax.set_ylabel("Revenue ($)", fontsize=10)
            ax.tick_params(axis="x", rotation=25)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"${int(height):,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    c_bot1, c_bot2 = st.columns(2)

    with c_bot1:
        st.subheader("🏆 Top 5 Best-Selling Products")
        top_prod_df = get_top_products(filtered_df, top_n=5)
        st.dataframe(
            top_prod_df.rename(columns={"Revenue": "Revenue ($)", "UnitsSold": "Units Sold"}),
            use_container_width=True,
            hide_index=True,
        )

    with c_bot2:
        st.subheader("🌍 Revenue by Top Countries")
        country_df = get_revenue_by_country(filtered_df).head(5)
        st.dataframe(
            country_df.rename(columns={"Revenue": "Revenue ($)", "Orders": "Total Orders"}),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# SECTION 2: SALES ANALYSIS
# =========================================================
elif nav_selection == "Sales Analysis":
    st.title("Sales Analysis")
    st.markdown("In-depth analysis of revenue, order volume, product trends, and regional performance.")
    st.markdown("---")

    render_kpis(filtered_df)
    st.markdown("---")

    # 1. Monthly Revenue & Orders
    st.subheader("1. Monthly Revenue & Order Volume Trends")
    monthly_df = get_monthly_revenue(filtered_df)
    if not monthly_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        # Monthly Revenue Line
        ax1.plot(monthly_df["YearMonth"], monthly_df["Revenue"], marker="s", color="#0D47A1", linewidth=2.5)
        ax1.set_title("Monthly Revenue ($)", fontweight="bold")
        ax1.set_xlabel("Year-Month")
        ax1.set_ylabel("Revenue ($)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Monthly Orders Bar
        ax2.bar(monthly_df["YearMonth"], monthly_df["Orders"], color="#26A69A", alpha=0.85)
        ax2.set_title("Monthly Order Volume", fontweight="bold")
        ax2.set_xlabel("Year-Month")
        ax2.set_ylabel("Distinct Orders")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 2. Daily Orders Distribution
    st.markdown("---")
    st.subheader("2. Daily Order Activity")
    daily_df = get_daily_orders(filtered_df)
    if not daily_df.empty:
        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.plot(daily_df["Date"], daily_df["Orders"], color="#5C6BC0", linewidth=1.5, alpha=0.8)
        ax.set_title("Daily Transactions Volume", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Orders per Day")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 3. Category & Country Breakdown
    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("3. Revenue Distribution by Category")
        cat_df = get_revenue_by_category(filtered_df)
        if not cat_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                cat_df["Revenue"],
                labels=cat_df["Category"],
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("pastel", len(cat_df)),
                wedgeprops=dict(width=0.6, edgecolor="w"),
            )
            ax.set_title("Category Revenue Share", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_c2:
        st.subheader("4. Revenue by Country")
        country_df = get_revenue_by_country(filtered_df)
        if not country_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(data=country_df, x="Revenue", y="Country", palette="viridis", ax=ax)
            ax.set_title("Sales Revenue by Market ($)", fontweight="bold")
            ax.set_xlabel("Revenue ($)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # 4. Top 10 Products
    st.markdown("---")
    st.subheader("5. Top 10 Best-Performing Products by Revenue")
    top_prods = get_top_products(filtered_df, top_n=10)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=top_prods, x="Revenue", y="Product", palette="Blues_r", ax=ax)
    ax.set_title("Top 10 Products by Total Revenue ($)", fontweight="bold")
    ax.set_xlabel("Revenue ($)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =========================================================
# SECTION 3: CUSTOMER ANALYSIS
# =========================================================
elif nav_selection == "Customer Analysis":
    st.title("Customer Analysis")
    st.markdown("Understand customer acquisition, purchase frequency, order volume, and lifetime value distribution.")
    st.markdown("---")

    # Metrics on New vs Returning
    repeat_metrics = get_new_vs_returning_customers(filtered_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{filtered_df['CustomerID'].nunique():,}")
    with c2:
        st.metric("One-Time Buyers", f"{repeat_metrics['single_order_customers']:,}")
    with c3:
        st.metric("Repeat Buyers", f"{repeat_metrics['repeat_customers']:,}")
    with c4:
        st.metric("Repeat Customer Rate", f"{repeat_metrics['repeat_rate_pct']}%")

    st.markdown("---")

    # Customer summary table
    st.subheader("Customer Summary Ranking")
    cust_table = get_customer_summary_table(filtered_df)

    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        min_spend_filter = st.number_input(
            "Filter Minimum Spending ($)", min_value=0.0, value=0.0, step=100.0
        )
    with col_t1:
        search_cust = st.text_input("Search Customer ID (e.g. CUST-1001)")

    filtered_cust_table = cust_table[cust_table["TotalSpending"] >= min_spend_filter]
    if search_cust:
        filtered_cust_table = filtered_cust_table[
            filtered_cust_table["CustomerID"].str.contains(search_cust.strip(), case=False)
        ]

    st.dataframe(filtered_cust_table.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Showing top {min(50, len(filtered_cust_table))} of {len(filtered_cust_table):,} customers matching criteria.")

    st.markdown("---")
    c_dist1, c_dist2 = st.columns(2)

    with c_dist1:
        st.subheader("Orders Per Customer Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(cust_table["TotalOrders"], bins=range(1, cust_table["TotalOrders"].max() + 2), kde=False, color="#1976D2", ax=ax)
        ax.set_title("Frequency Distribution (Orders Count)", fontweight="bold")
        ax.set_xlabel("Number of Orders")
        ax.set_ylabel("Customer Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c_dist2:
        st.subheader("Customer Spending Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(cust_table["TotalSpending"], bins=20, kde=True, color="#388E3C", ax=ax)
        ax.set_title("Monetary Distribution (Total Spend $)", fontweight="bold")
        ax.set_xlabel("Total Spend ($)")
        ax.set_ylabel("Customer Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# =========================================================
# SECTION 4: RFM SEGMENTATION
# =========================================================
elif nav_selection == "RFM Segmentation":
    st.title("RFM Analysis & Customer Segmentation")
    st.markdown(
        "**RFM Analysis** is an industry-standard behavioral marketing framework that groups customers "
        "based on three dimensions: **Recency** (how recently they bought), **Frequency** (how often they buy), "
        "and **Monetary Value** (how much they spend)."
    )

    # Explanation expander
    with st.expander("ℹ️ What do the RFM Metrics mean?", expanded=False):
        c_rfm1, c_rfm2, c_rfm3 = st.columns(3)
        with c_rfm1:
            st.markdown(f"**Recency (R)**: {RFM_METRIC_DESCRIPTIONS['Recency']}")
        with c_rfm2:
            st.markdown(f"**Frequency (F)**: {RFM_METRIC_DESCRIPTIONS['Frequency']}")
        with c_rfm3:
            st.markdown(f"**Monetary (M)**: {RFM_METRIC_DESCRIPTIONS['Monetary']}")

    st.markdown("---")

    # Compute RFM
    rfm_df = calculate_rfm(filtered_df)
    seg_summary = get_segment_summary(rfm_df)

    st.subheader("Customer Segment Summary")
    st.dataframe(
        seg_summary.rename(
            columns={
                "CustomerCount": "Customers",
                "CustomerSharePct": "Cust Share (%)",
                "TotalRevenue": "Revenue ($)",
                "RevenueSharePct": "Rev Share (%)",
                "AvgRecency": "Avg Recency (Days)",
                "AvgFrequency": "Avg Orders",
                "AvgMonetary": "Avg Spend ($)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Customer Distribution by Segment")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = sns.color_palette("Set2", len(seg_summary))
        sns.barplot(data=seg_summary, x="CustomerCount", y="Segment", palette="Blues_d", ax=ax)
        ax.set_title("Customer Count per Segment", fontweight="bold")
        ax.set_xlabel("Number of Customers")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_s2:
        st.subheader("Revenue Contribution by Segment")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=seg_summary, x="TotalRevenue", y="Segment", palette="Greens_d", ax=ax)
        ax.set_title("Total Revenue Generated ($)", fontweight="bold")
        ax.set_xlabel("Revenue ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Customer Segment Definitions & Action Playbook")
    for seg_name, desc in SEGMENT_DESCRIPTIONS.items():
        st.markdown(f"- **{seg_name}**: {desc}")

    st.markdown("---")
    st.subheader("Individual RFM Scores Table")
    selected_seg_filter = st.selectbox(
        "Filter by Segment", ["All Segments"] + seg_summary["Segment"].tolist()
    )
    if selected_seg_filter != "All Segments":
        display_rfm = rfm_df[rfm_df["Segment"] == selected_seg_filter]
    else:
        display_rfm = rfm_df

    st.dataframe(
        display_rfm[["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]].head(50),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# SECTION 5: K-MEANS CLUSTERING
# =========================================================
elif nav_selection == "K-Means Clustering":
    st.title("Machine Learning: K-Means Customer Clustering")
    st.markdown(
        "Unsupervised Machine Learning using **K-Means Clustering** groups customers mathematically "
        "by identifying clusters in standardized Recency, Frequency, and Monetary space."
    )
    st.markdown("---")

    rfm_df = calculate_rfm(filtered_df)

    col_km_ctrl1, col_km_ctrl2 = st.columns([1, 2])
    with col_km_ctrl1:
        st.subheader("Cluster Configuration")
        k_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=4)
        run_clustering = st.button("Re-compute K-Means Clustering", use_container_width=True)

    with col_km_ctrl2:
        st.subheader("Elbow Method & Silhouette Evaluation")
        with st.spinner("Computing Elbow and Silhouette metrics..."):
            elbow_data = compute_elbow_and_silhouette(rfm_df, min_k=2, max_k=6)

        fig, (ax_elb, ax_sil) = plt.subplots(1, 2, figsize=(10, 3.8))
        ax_elb.plot(elbow_data["k_values"], elbow_data["inertia_values"], marker="o", color="#E65100", linewidth=2)
        ax_elb.set_title("Elbow Method (Inertia)", fontweight="bold")
        ax_elb.set_xlabel("Number of Clusters (k)")
        ax_elb.set_ylabel("WCSS / Inertia")

        ax_sil.plot(elbow_data["k_values"], elbow_data["silhouette_scores"], marker="s", color="#1565C0", linewidth=2)
        ax_sil.set_title("Silhouette Scores", fontweight="bold")
        ax_sil.set_xlabel("Number of Clusters (k)")
        ax_sil.set_ylabel("Silhouette Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader(f"K-Means Clustering Results (k={k_clusters})")

    clustered_df, kmeans_model, scaler, pca = perform_kmeans_clustering(rfm_df, n_clusters=k_clusters)
    cluster_profiles = get_cluster_profiles(clustered_df)

    col_v1, col_v2 = st.columns([1.2, 1])

    with col_v1:
        st.markdown("#### 2D PCA Projection of Clusters")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=clustered_df,
            x="PCA1",
            y="PCA2",
            hue="Cluster",
            palette="tab10",
            s=60,
            alpha=0.85,
            ax=ax,
        )
        ax.set_title(f"Customer Clusters in PCA Feature Space (k={k_clusters})", fontweight="bold")
        ax.set_xlabel("Principal Component 1 (PCA1)")
        ax.set_ylabel("Principal Component 2 (PCA2)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_v2:
        st.markdown("#### Cluster Interpretation & Profiles")
        st.dataframe(
            cluster_profiles[["Cluster", "CustomerCount", "AvgRecency", "AvgFrequency", "AvgMonetary", "ClusterPersona"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Business Explanation of Clusters")
    for _, row in cluster_profiles.iterrows():
        st.markdown(
            f"- **Cluster {int(row['Cluster'])}**: {row['ClusterPersona']}. "
            f"Averages: {row['AvgRecency']} days recency, {row['AvgFrequency']} orders, ${row['AvgMonetary']:,.2f} total spend."
        )


# =========================================================
# SECTION 6: CHURN PREDICTION
# =========================================================
elif nav_selection == "Churn Prediction":
    st.title("Machine Learning: Customer Churn Prediction")
    st.markdown(
        "A supervised classification pipeline using **Logistic Regression** to identify customers "
        "at risk of churning based on their purchase habits and demographics."
    )

    st.info(
        "📌 **Methodology Note**: In this sample project, churn is labeled using an inactivity threshold "
        "(e.g., customers with no purchase in the last 90 days). To adhere to ML best practices and avoid "
        "data leakage, the predictive model trains strictly on behavioral features (Frequency, Monetary Spend, "
        "Average Order Value, and Customer Age) with strict train/test data separation before scaling."
    )
    st.markdown("---")

    rfm_df = calculate_rfm(filtered_df)
    churn_dataset = prepare_churn_data(rfm_df, customer_demographics=filtered_df, inactivity_threshold_days=90)

    # Train / Evaluate Model
    with st.spinner("Training and evaluating churn model..."):
        pipeline, metrics, test_data = train_churn_model(
            churn_dataset, model_save_path=MODEL_PATH
        )

    # Metric Cards
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    c_m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    c_m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    c_m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    c_m4.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    c_m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("---")
    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.subheader("Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Active (0)", "Churned (1)"],
            yticklabels=["Active (0)", "Churned (1)"],
            ax=ax,
        )
        ax.set_title("Test Set Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_eval2:
        st.subheader("Model Interpretation")
        lr_coefs = pipeline.named_steps["classifier"].coef_[0]
        coef_df = pd.DataFrame({
            "Feature": metrics["feature_names"],
            "Coefficient": lr_coefs,
        }).sort_values("Coefficient", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm", ax=ax)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_title("Feature Weights (Log-Odds Impact on Churn)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔮 Interactive Customer Churn Risk Calculator")
    st.markdown("Test the model in real time for a hypothetical or specific customer profile:")

    calc_c1, calc_c2, calc_c3, calc_c4 = st.columns(4)
    with calc_c1:
        in_freq = st.number_input("Total Orders (Frequency)", min_value=1, max_value=50, value=2)
    with calc_c2:
        in_mon = st.number_input("Total Spending ($)", min_value=10.0, max_value=10000.0, value=150.0)
    with calc_c3:
        in_aov = st.number_input("Average Order Value ($)", min_value=5.0, max_value=2000.0, value=75.0)
    with calc_c4:
        in_age = st.slider("Customer Age", min_value=18, max_value=75, value=35)

    test_customer = pd.DataFrame([{
        "Frequency": in_freq,
        "Monetary": in_mon,
        "AverageOrderValue": in_aov,
        "CustomerAge": in_age,
    }])

    pred_res = predict_customer_churn(test_customer, model_path=MODEL_PATH)
    prob_churn = pred_res["ChurnProbability"].iloc[0]
    risk_lvl = pred_res["RiskLevel"].iloc[0]

    st.markdown(
        f"**Model Assessment:** Predicted Churn Probability = **{prob_churn * 100:.1f}%** "
        f"| Risk Tier = **{risk_lvl}**"
    )
    if prob_churn >= 0.5:
        st.warning(f"⚠️ High Churn Risk! Proactive retention offer or feedback survey recommended.")
    else:
        st.success(f"✅ Healthy Customer Activity! Candidate for cross-sell recommendations.")


# =========================================================
# SECTION 7: BUSINESS INSIGHTS
# =========================================================
elif nav_selection == "Business Insights":
    st.title("Business Insights & Strategic Recommendations")
    st.markdown(
        "Executive summary highlighting key performance drivers, customer concentrations, "
        "and data-backed business recommendations."
    )
    st.markdown("---")

    rfm_df = calculate_rfm(filtered_df)
    insights_data = generate_business_insights(filtered_df, rfm_df)
    ins = insights_data["insights"]

    # Highlights cards
    c_i1, c_i2, c_i3 = st.columns(3)
    with c_i1:
        st.info(
            f"**Top Revenue Category**\n### {ins['top_category']}\n"
            f"Revenue: ${ins['top_category_revenue']:,.2f} ({ins['top_category_share']}% share)"
        )
    with c_i2:
        st.success(
            f"**Best Performing Product**\n### {ins['best_product']}\n"
            f"Sales: ${ins['best_product_revenue']:,.2f} ({ins['best_product_units']} units)"
        )
    with c_i3:
        st.warning(
            f"**At-Risk Customer Base**\n### {ins['at_risk_customers']} Customers\n"
            f"{ins['at_risk_pct']}% of customer base inactive > 90 days"
        )

    st.markdown("---")
    st.subheader("Key Findings & Drivers")
    st.markdown(
        f"- **Pareto Concentration**: The top 10% of customers account for **{ins['top_10_pct_customer_share']}%** of all revenue generated.\n"
        f"- **Primary Customer Pillar**: The **{ins['top_segment']}** segment generates **{ins['top_segment_share']}%** of gross store sales.\n"
        f"- **Recent Momentum**: Month-over-Month store revenue trend shifted by **{ins['mom_growth_pct']}%** in the most recent completed period."
    )

    st.markdown("---")
    st.subheader("Data-Backed Strategic Recommendations")
    for rec in insights_data["recommendations"]:
        st.markdown(f"- {rec}")


# =========================================================
# SECTION 8: SQL EXPLORER
# =========================================================
elif nav_selection == "SQL Explorer":
    st.title("SQLite Database Query Explorer")
    st.markdown(
        "Execute relational SQL queries directly on the SQLite database (`data/ecommerce.db`) "
        "powering this application. Table name: `ecommerce_orders`."
    )
    st.markdown("---")

    # Sample canned queries
    sql_presets = {
        "1. Total Revenue": "SELECT ROUND(SUM(TotalAmount), 2) AS total_revenue FROM ecommerce_orders;",
        "2. Total Orders & Customers": "SELECT COUNT(DISTINCT OrderID) AS total_orders, COUNT(DISTINCT CustomerID) AS total_customers FROM ecommerce_orders;",
        "3. Revenue by Category": "SELECT Category, ROUND(SUM(TotalAmount), 2) AS category_revenue, SUM(Quantity) AS units_sold FROM ecommerce_orders GROUP BY Category ORDER BY category_revenue DESC;",
        "4. Top 10 Best Selling Products": "SELECT Product, Category, ROUND(SUM(TotalAmount), 2) AS total_revenue, SUM(Quantity) AS units_sold FROM ecommerce_orders GROUP BY Product, Category ORDER BY total_revenue DESC LIMIT 10;",
        "5. Top 10 Most Valuable Customers": "SELECT CustomerID, Country, COUNT(DISTINCT OrderID) AS total_orders, ROUND(SUM(TotalAmount), 2) AS total_spent FROM ecommerce_orders GROUP BY CustomerID, Country ORDER BY total_spent DESC LIMIT 10;",
        "6. Monthly Revenue Trend": "SELECT strftime('%Y-%m', OrderDate) AS year_month, ROUND(SUM(TotalAmount), 2) AS monthly_revenue, COUNT(DISTINCT OrderID) AS orders FROM ecommerce_orders GROUP BY year_month ORDER BY year_month ASC;",
        "7. Overall Average Order Value (AOV)": "SELECT ROUND(SUM(TotalAmount) / COUNT(DISTINCT OrderID), 2) AS avg_order_value FROM ecommerce_orders;",
        "8. High-Spending Customers (> $1500)": "SELECT CustomerID, COUNT(DISTINCT OrderID) AS total_orders, ROUND(SUM(TotalAmount), 2) AS total_spent FROM ecommerce_orders GROUP BY CustomerID HAVING total_spent > 1500 ORDER BY total_spent DESC;",
        "9. Inactive Customers (> 90 Days)": "SELECT CustomerID, MAX(OrderDate) AS last_order_date, CAST((julianday((SELECT MAX(OrderDate) FROM ecommerce_orders)) - julianday(MAX(OrderDate))) AS INTEGER) AS days_inactive FROM ecommerce_orders GROUP BY CustomerID HAVING days_inactive > 90 ORDER BY days_inactive DESC LIMIT 20;",
    }

    selected_preset = st.selectbox("Choose a Pre-Built Analytical Query:", list(sql_presets.keys()))
    default_query_text = sql_presets[selected_preset]

    query_input = st.text_area("SQL Query Editor (SQLite syntax)", value=default_query_text, height=120)

    if st.button("Run SQL Query", type="primary"):
        try:
            query_res = execute_sql_query(query_input, DB_PATH)
            st.success(f"Query returned {len(query_res)} rows.")
            st.dataframe(query_res, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")


# =========================================================
# SECTION 9: DATA PREVIEW & CLEANING STATS
# =========================================================
elif nav_selection == "Data Preview & Cleaning Stats":
    st.title("Data Preview & Cleaning Pipeline Statistics")
    st.markdown("Audit transparency for dataset preprocessing, anomaly handling, and schema validation.")
    st.markdown("---")

    if cleaning_stats:
        st.subheader("Data Cleaning Audit Metrics")
        cs1, cs2, cs3, cs4, cs5 = st.columns(5)
        cs1.metric("Original Rows", f"{cleaning_stats['original_rows']:,}")
        cs2.metric("Duplicates Removed", f"{cleaning_stats['removed_duplicates']:,}")
        cs3.metric("Missing IDs Dropped", f"{cleaning_stats['missing_customer_ids']:,}")
        cs4.metric("Invalid Values Removed", f"{cleaning_stats['invalid_values_removed']:,}")
        cs5.metric("Final Clean Rows", f"{cleaning_stats['final_rows']:,}")

    st.markdown("---")
    st.subheader("Cleaned Dataset Preview (First 50 Rows)")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Schema & Data Types")
    dtypes_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": filtered_df.dtypes.astype(str),
        "Non-Null Count": filtered_df.notnull().sum().values,
        "Unique Values": filtered_df.nunique().values,
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "E-Commerce Customer Analytics | Portfolio Project | Built with Streamlit, Python & Scikit-learn"
)
