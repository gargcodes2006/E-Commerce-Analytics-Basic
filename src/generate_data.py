"""
Data Generator Script for Sample E-Commerce Dataset
This script generates a realistic sample dataset for development, testing,
and portfolio demonstration. All data is synthetically generated sample data.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_ecommerce_data(
    num_records: int = 4000,
    num_customers: int = 600,
    random_seed: int = 42,
    output_path: str = "data/ecommerce_data.csv",
) -> pd.DataFrame:
    """Generate realistic e-commerce transactions dataset with controlled noise

    for data cleaning demonstration.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Catalog definition
    catalog = [
        {"Product": "4K Ultra HD Monitor", "Category": "Electronics", "UnitPrice": 329.99},
        {"Product": "Smart Watch Pro", "Category": "Electronics", "UnitPrice": 189.99},
        {"Product": "10-inch Tablet", "Category": "Electronics", "UnitPrice": 249.99},
        {"Product": "Ergonomic Wireless Mouse", "Category": "Computer Peripherals", "UnitPrice": 34.99},
        {"Product": "Mechanical Keyboard RGB", "Category": "Computer Peripherals", "UnitPrice": 89.99},
        {"Product": "USB-C Multiport Hub", "Category": "Computer Peripherals", "UnitPrice": 44.99},
        {"Product": "HD Pro Webcam 1080p", "Category": "Computer Peripherals", "UnitPrice": 74.99},
        {"Product": "Noise-Cancelling Headphones", "Category": "Audio", "UnitPrice": 199.99},
        {"Product": "Bluetooth Portable Speaker", "Category": "Audio", "UnitPrice": 59.99},
        {"Product": "Wireless Sport Earbuds", "Category": "Audio", "UnitPrice": 79.99},
        {"Product": "Ergonomic Desk Chair", "Category": "Home & Office", "UnitPrice": 239.99},
        {"Product": "Adjustable Laptop Stand", "Category": "Home & Office", "UnitPrice": 39.99},
        {"Product": "Dimmable LED Desk Lamp", "Category": "Home & Office", "UnitPrice": 29.99},
        {"Product": "Cable Management Kit", "Category": "Home & Office", "UnitPrice": 17.99},
        {"Product": "External SSD 1TB", "Category": "Accessories", "UnitPrice": 109.99},
        {"Product": "Waterproof Laptop Backpack", "Category": "Accessories", "UnitPrice": 54.99},
        {"Product": "Fast Wireless Charger Pad", "Category": "Accessories", "UnitPrice": 27.99},
    ]

    countries = ["United States", "United Kingdom", "Germany", "Canada", "Australia", "France"]
    country_weights = [0.45, 0.20, 0.12, 0.10, 0.08, 0.05]

    genders = ["Male", "Female", "Other"]
    gender_weights = [0.49, 0.48, 0.03]

    # Pre-generate customer profiles
    customers = []
    for i in range(1, num_customers + 1):
        cust_id = f"CUST-{1000 + i}"
        country = random.choices(countries, weights=country_weights)[0]
        gender = random.choices(genders, weights=gender_weights)[0]
        age = int(np.clip(np.random.normal(36, 12), 18, 72))
        # Customer activity bias (some purchase heavily, some rarely)
        activity_weight = float(np.random.pareto(a=1.8) + 0.2)
        customers.append({
            "CustomerID": cust_id,
            "Country": country,
            "Gender": gender,
            "CustomerAge": age,
            "Weight": activity_weight,
        })

    cust_weights = [c["Weight"] for c in customers]
    norm_cust_weights = [w / sum(cust_weights) for w in cust_weights]

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 3, 31)
    date_range_days = (end_date - start_date).days

    records = []
    order_counter = 1000

    # Generate base records
    for _ in range(num_records):
        # Pick customer
        selected_cust = random.choices(customers, weights=norm_cust_weights)[0]
        
        # Pick date (slight growth over time and seasonality)
        day_offset = int(np.random.beta(a=1.3, b=1.0) * date_range_days)
        order_date = start_date + timedelta(days=day_offset)
        
        # Pick product
        item = random.choice(catalog)
        
        # Order ID - reuse or create new
        if random.random() < 0.35 and len(records) > 0 and records[-1]["CustomerID"] == selected_cust["CustomerID"]:
            order_id = records[-1]["OrderID"]
            order_date_str = records[-1]["OrderDate"]
        else:
            order_counter += 1
            order_id = f"ORD-{order_counter}"
            order_date_str = order_date.strftime("%Y-%m-%d")

        # Quantity: mostly 1-3, occasionally larger
        quantity = int(np.random.choice([1, 2, 3, 4, 5], p=[0.60, 0.22, 0.10, 0.05, 0.03]))
        unit_price = item["UnitPrice"]
        total_amount = round(quantity * unit_price, 2)

        records.append({
            "OrderID": order_id,
            "CustomerID": selected_cust["CustomerID"],
            "OrderDate": order_date_str,
            "Product": item["Product"],
            "Category": item["Category"],
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "TotalAmount": total_amount,
            "Country": selected_cust["Country"],
            "CustomerAge": selected_cust["CustomerAge"],
            "Gender": selected_cust["Gender"],
        })

    df = pd.DataFrame(records)

    # Introduce deliberate realistic anomalies for data cleaning demonstration:
    # 1. Missing CustomerIDs (e.g. guest checkouts) ~25 rows
    guest_idx = random.sample(range(len(df)), 25)
    df.loc[guest_idx, "CustomerID"] = np.nan

    # 2. Negative quantities (returns or errors) ~15 rows
    neg_idx = random.sample(range(len(df)), 15)
    df.loc[neg_idx, "Quantity"] = -1 * df.loc[neg_idx, "Quantity"]
    df.loc[neg_idx, "TotalAmount"] = -1 * abs(df.loc[neg_idx, "TotalAmount"])

    # 3. Missing UnitPrice ~10 rows
    missing_price_idx = random.sample(range(len(df)), 10)
    df.loc[missing_price_idx, "UnitPrice"] = np.nan

    # 4. Missing CustomerAge ~15 rows
    missing_age_idx = random.sample(range(len(df)), 15)
    df.loc[missing_age_idx, "CustomerAge"] = np.nan

    # 5. Duplicate rows ~20 rows
    duplicate_rows = df.sample(n=20, random_state=random_seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Save to destination
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset with {len(df)} rows saved to {output_path}")
    return df


if __name__ == "__main__":
    generate_ecommerce_data()
