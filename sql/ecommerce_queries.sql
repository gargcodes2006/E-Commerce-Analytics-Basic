-- ==============================================================================
-- E-Commerce Customer Analytics - Core SQL Queries (SQLite Compatible)
-- Table: ecommerce_orders
-- Columns: OrderID, CustomerID, OrderDate, Product, Category, Quantity, UnitPrice,
--          TotalAmount, Country, CustomerAge, Gender
-- ==============================================================================

-- 1. Total Revenue across all transactions
SELECT 
    ROUND(SUM(TotalAmount), 2) AS total_revenue
FROM 
    ecommerce_orders;


-- 2. Total Orders (distinct transactions count)
SELECT 
    COUNT(DISTINCT OrderID) AS total_orders
FROM 
    ecommerce_orders;


-- 3. Total Customers (unique buyers)
SELECT 
    COUNT(DISTINCT CustomerID) AS total_customers
FROM 
    ecommerce_orders;


-- 4. Revenue and units sold broken down by Product Category
SELECT 
    Category,
    ROUND(SUM(TotalAmount), 2) AS total_revenue,
    SUM(Quantity) AS units_sold,
    COUNT(DISTINCT OrderID) AS total_orders
FROM 
    ecommerce_orders
GROUP BY 
    Category
ORDER BY 
    total_revenue DESC;


-- 5. Top 10 Best-Selling Products by Revenue
SELECT 
    Product,
    Category,
    ROUND(SUM(TotalAmount), 2) AS total_revenue,
    SUM(Quantity) AS units_sold,
    ROUND(AVG(UnitPrice), 2) AS avg_unit_price
FROM 
    ecommerce_orders
GROUP BY 
    Product, 
    Category
ORDER BY 
    total_revenue DESC
LIMIT 10;


-- 6. Top 10 Most Valuable Customers by Total Spending
SELECT 
    CustomerID,
    Country,
    COUNT(DISTINCT OrderID) AS total_orders,
    ROUND(SUM(TotalAmount), 2) AS total_spending,
    ROUND(SUM(TotalAmount) / COUNT(DISTINCT OrderID), 2) AS avg_order_value
FROM 
    ecommerce_orders
GROUP BY 
    CustomerID,
    Country
ORDER BY 
    total_spending DESC
LIMIT 10;


-- 7. Monthly Revenue and Order Volume Trends
SELECT 
    strftime('%Y-%m', OrderDate) AS year_month,
    ROUND(SUM(TotalAmount), 2) AS monthly_revenue,
    COUNT(DISTINCT OrderID) AS monthly_orders,
    COUNT(DISTINCT CustomerID) AS active_customers
FROM 
    ecommerce_orders
GROUP BY 
    year_month
ORDER BY 
    year_month ASC;


-- 8. Overall Average Order Value (AOV)
SELECT 
    ROUND(SUM(TotalAmount) / COUNT(DISTINCT OrderID), 2) AS average_order_value
FROM 
    ecommerce_orders;


-- 9. High-Value Customers (Total spending exceeding $1,500)
SELECT 
    CustomerID,
    COUNT(DISTINCT OrderID) AS total_orders,
    ROUND(SUM(TotalAmount), 2) AS total_spent,
    ROUND(AVG(TotalAmount), 2) AS avg_item_amount
FROM 
    ecommerce_orders
GROUP BY 
    CustomerID
HAVING 
    total_spent > 1500
ORDER BY 
    total_spent DESC;


-- 10. Inactive Customers (No purchase in over 90 days relative to latest store date)
SELECT 
    CustomerID,
    MAX(OrderDate) AS last_purchase_date,
    CAST((julianday((SELECT MAX(OrderDate) FROM ecommerce_orders)) - julianday(MAX(OrderDate))) AS INTEGER) AS days_inactive,
    COUNT(DISTINCT OrderID) AS total_orders,
    ROUND(SUM(TotalAmount), 2) AS total_spent
FROM 
    ecommerce_orders
GROUP BY 
    CustomerID
HAVING 
    days_inactive > 90
ORDER BY 
    days_inactive DESC;
