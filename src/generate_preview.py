"""
Generate a high-resolution dashboard preview image representing the
Streamlit application for GitHub README and documentation.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.data_cleaning import load_and_clean_data
from src.analysis import get_monthly_revenue, get_revenue_by_category, compute_kpis
from src.rfm_analysis import calculate_rfm, get_segment_summary
from src.customer_segmentation import perform_kmeans_clustering
from src.churn_prediction import prepare_churn_data, train_churn_model

os.makedirs("images", exist_ok=True)
preview_path = os.path.join("images", "dashboard_preview.png")

# Load data and run models
df, _ = load_and_clean_data()
kpis = compute_kpis(df)
rfm = calculate_rfm(df)
seg_summary = get_segment_summary(rfm)
clustered_df, _, _, _ = perform_kmeans_clustering(rfm, n_clusters=4)
churn_df = prepare_churn_data(rfm, df)
_, metrics, _ = train_churn_model(churn_df)

# Create multi-panel dashboard preview
fig = plt.figure(figsize=(16, 10), facecolor="#F8FAFC")
gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.28)

# Header Title Block
fig.text(0.08, 0.95, "E-Commerce Customer Analytics Dashboard", fontsize=18, fontweight="bold", color="#0F172A")
fig.text(0.08, 0.925, f"KPIs: Revenue: ${kpis['total_revenue']:,.2f}  |  Orders: {kpis['total_orders']:,}  |  Customers: {kpis['total_customers']:,}  |  AOV: ${kpis['avg_order_value']:.2f}",
         fontsize=11, color="#475569")

# Panel 1: Monthly Revenue Trend
ax1 = fig.add_subplot(gs[0, :2])
monthly = get_monthly_revenue(df)
ax1.plot(monthly["YearMonth"], monthly["Revenue"], marker="o", color="#2563EB", linewidth=2.5)
ax1.fill_between(monthly["YearMonth"], monthly["Revenue"], alpha=0.12, color="#2563EB")
ax1.set_title("Monthly Revenue Trajectory ($)", fontweight="bold", color="#1E293B", pad=8)
ax1.tick_params(axis="x", rotation=35, labelsize=8)
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: Revenue by Category
ax2 = fig.add_subplot(gs[0, 2])
cat_rev = get_revenue_by_category(df)
sns.barplot(data=cat_rev, x="Revenue", y="Category", palette="Blues_r", ax=ax2)
ax2.set_title("Revenue by Category ($)", fontweight="bold", color="#1E293B", pad=8)
ax2.set_xlabel("Revenue ($)", fontsize=9)
ax2.tick_params(labelsize=8)

# Panel 3: RFM Customer Segments Share
ax3 = fig.add_subplot(gs[1, 0])
sns.barplot(data=seg_summary, x="CustomerCount", y="Segment", palette="Set2", ax=ax3)
ax3.set_title("RFM Customer Segments", fontweight="bold", color="#1E293B", pad=8)
ax3.set_xlabel("Customers", fontsize=9)
ax3.tick_params(labelsize=8)

# Panel 4: RFM Revenue Contribution
ax4 = fig.add_subplot(gs[1, 1])
sns.barplot(data=seg_summary, x="TotalRevenue", y="Segment", palette="Greens_r", ax=ax4)
ax4.set_title("Revenue by RFM Segment ($)", fontweight="bold", color="#1E293B", pad=8)
ax4.set_xlabel("Revenue ($)", fontsize=9)
ax4.tick_params(labelsize=8)

# Panel 5: Churn Confusion Matrix
ax5 = fig.add_subplot(gs[1, 2])
cm = np.array(metrics["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Active", "Churned"], yticklabels=["Active", "Churned"], ax=ax5)
ax5.set_title(f"Churn Model (ROC-AUC: {metrics['roc_auc']:.2f})", fontweight="bold", color="#1E293B", pad=8)
ax5.set_xlabel("Predicted", fontsize=9)
ax5.set_ylabel("Actual", fontsize=9)
ax5.tick_params(labelsize=8)

# Panel 6: K-Means PCA 2D Clustering
ax6 = fig.add_subplot(gs[2, :])
sns.scatterplot(
    data=clustered_df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="tab10",
    s=65,
    alpha=0.85,
    ax=ax6
)
ax6.set_title("K-Means Behavioral Customer Clusters (2D PCA Space, k=4)", fontweight="bold", color="#1E293B", pad=8)
ax6.set_xlabel("Principal Component 1 (PCA1)", fontsize=9)
ax6.set_ylabel("Principal Component 2 (PCA2)", fontsize=9)
ax6.legend(title="Cluster", loc="upper right", fontsize=8)
ax6.grid(True, linestyle="--", alpha=0.5)

plt.savefig(preview_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"Dashboard preview saved to: {preview_path}")
