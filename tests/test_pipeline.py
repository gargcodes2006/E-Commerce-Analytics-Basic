"""
Automated End-to-End Test Suite for E-Commerce Customer Analytics.
Validates all modules, queries, models, and dashboard components.
"""

import os
import sys
import unittest
import sqlite3
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from src.rfm_analysis import calculate_rfm, get_segment_summary
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


class TestEcommerceAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(PROJECT_ROOT, "data", "ecommerce_data.csv")
        cls.db_path = os.path.join(PROJECT_ROOT, "data", "ecommerce.db")
        cls.model_path = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")

    def test_01_data_cleaning(self):
        """Test data cleaning pipeline and audit statistics."""
        df_clean, stats = load_and_clean_data(self.data_path)
        self.assertFalse(df_clean.empty)
        self.assertGreater(stats["original_rows"], stats["final_rows"])
        self.assertGreater(stats["removed_duplicates"], 0)
        self.assertGreater(stats["missing_customer_ids"], 0)
        self.assertFalse(df_clean["CustomerID"].isnull().any())
        self.assertTrue((df_clean["Quantity"] > 0).all())
        self.assertTrue((df_clean["UnitPrice"] > 0).all())

    def test_02_analysis_kpis_and_trends(self):
        """Test sales and customer analysis aggregations."""
        df_clean, _ = load_and_clean_data(self.data_path)
        kpis = compute_kpis(df_clean)
        self.assertGreater(kpis["total_revenue"], 0)
        self.assertGreater(kpis["total_orders"], 0)
        self.assertGreater(kpis["total_customers"], 0)
        self.assertGreater(kpis["avg_order_value"], 0)

        monthly = get_monthly_revenue(df_clean)
        self.assertGreater(len(monthly), 1)

        daily = get_daily_orders(df_clean)
        self.assertFalse(daily.empty)

        cat_rev = get_revenue_by_category(df_clean)
        self.assertEqual(len(cat_rev), 5)

        top_prods = get_top_products(df_clean, top_n=10)
        self.assertEqual(len(top_prods), 10)

        country_rev = get_revenue_by_country(df_clean)
        self.assertGreater(len(country_rev), 1)

        cust_table = get_customer_summary_table(df_clean)
        self.assertEqual(len(cust_table), kpis["total_customers"])

        repeat_stats = get_new_vs_returning_customers(df_clean)
        self.assertIn("repeat_rate_pct", repeat_stats)

    def test_03_rfm_analysis(self):
        """Test RFM score calculation and customer segments."""
        df_clean, _ = load_and_clean_data(self.data_path)
        rfm = calculate_rfm(df_clean)
        self.assertEqual(len(rfm), df_clean["CustomerID"].nunique())
        self.assertTrue(set(rfm["R_Score"].unique()).issubset({1, 2, 3, 4, 5}))
        self.assertTrue(set(rfm["F_Score"].unique()).issubset({1, 2, 3, 4, 5}))
        self.assertTrue(set(rfm["M_Score"].unique()).issubset({1, 2, 3, 4, 5}))

        expected_segments = {
            "Champions", "Loyal Customers", "Potential Loyalists",
            "New Customers", "At Risk", "Lost Customers"
        }
        self.assertTrue(set(rfm["Segment"].unique()).issubset(expected_segments))

        summary = get_segment_summary(rfm)
        self.assertEqual(summary["CustomerCount"].sum(), len(rfm))

    def test_04_kmeans_segmentation(self):
        """Test K-Means clustering, elbow method, and PCA 2D coordinates."""
        df_clean, _ = load_and_clean_data(self.data_path)
        rfm = calculate_rfm(df_clean)

        elbow = compute_elbow_and_silhouette(rfm, min_k=2, max_k=5)
        self.assertEqual(len(elbow["k_values"]), 4)
        self.assertEqual(len(elbow["silhouette_scores"]), 4)

        clustered, km, scaler, pca = perform_kmeans_clustering(rfm, n_clusters=4)
        self.assertIn("Cluster", clustered.columns)
        self.assertIn("PCA1", clustered.columns)
        self.assertIn("PCA2", clustered.columns)
        self.assertEqual(len(clustered["Cluster"].unique()), 4)

        profiles = get_cluster_profiles(clustered)
        self.assertEqual(len(profiles), 4)

    def test_05_churn_prediction(self):
        """Test churn dataset preparation, logistic regression training, and inference."""
        df_clean, _ = load_and_clean_data(self.data_path)
        rfm = calculate_rfm(df_clean)
        churn_df = prepare_churn_data(rfm, customer_demographics=df_clean, inactivity_threshold_days=90)
        self.assertIn("IsChurned", churn_df.columns)

        pipeline, metrics, test_data = train_churn_model(
            churn_df, model_save_path=self.model_path
        )
        self.assertTrue(os.path.exists(self.model_path))
        self.assertGreater(metrics["accuracy"], 0.5)
        self.assertGreater(metrics["recall"], 0.5)
        self.assertGreater(metrics["roc_auc"], 0.6)

        # Inference test
        sample_input = pd.DataFrame([{
            "Frequency": 3,
            "Monetary": 250.0,
            "AverageOrderValue": 83.33,
            "CustomerAge": 38,
        }])
        pred = predict_customer_churn(sample_input, model_path=self.model_path)
        self.assertIn("PredictedChurn", pred.columns)
        self.assertIn("ChurnProbability", pred.columns)
        self.assertIn("RiskLevel", pred.columns)

    def test_06_sql_queries(self):
        """Test all 10 SQLite queries from sql/ecommerce_queries.sql."""
        df_clean, _ = load_and_clean_data(self.data_path)
        init_sqlite_db(df_clean, self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

        sql_file = os.path.join(PROJECT_ROOT, "sql", "ecommerce_queries.sql")
        with open(sql_file, "r") as f:
            lines = f.readlines()
        clean_sql = "".join([l for l in lines if not l.strip().startswith("--")])
        queries = [q.strip() for q in clean_sql.split(";") if q.strip()]
        self.assertEqual(len(queries), 10)

        for idx, query in enumerate(queries, 1):
            res = execute_sql_query(query, self.db_path)
            self.assertFalse(res.empty, f"Query {idx} returned empty result")

    def test_07_business_insights(self):
        """Test generation of business insights and actionable recommendations."""
        df_clean, _ = load_and_clean_data(self.data_path)
        rfm = calculate_rfm(df_clean)
        insights = generate_business_insights(df_clean, rfm)
        self.assertIn("insights", insights)
        self.assertIn("recommendations", insights)
        self.assertGreater(len(insights["recommendations"]), 0)
        self.assertGreater(insights["insights"]["top_category_revenue"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
