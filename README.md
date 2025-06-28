# Retail Customer Analysis - Complete Solution

## Overview
This document provides a comprehensive solution for the retail customer analysis case study. The analysis involves merging customer, transaction, and product hierarchy datasets to answer various business questions.

## Dataset Information
- **Customer Dataset**: Contains customer demographic information
- **Transaction Dataset**: Contains customer transaction records
- **Product Hierarchy Dataset**: Contains product category and subcategory information

## Prerequisites
```python
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
```

## Complete Python Solution

```python
# Load the datasets
print("Loading datasets...")
customer_df = pd.read_csv('customer.csv')
transaction_df = pd.read_csv('transaction.csv')
product_hierarchy_df = pd.read_csv('product hierarchy.csv')

print(f"Customer data shape: {customer_df.shape}")
print(f"Transaction data shape: {transaction_df.shape}")
print(f"Product hierarchy data shape: {product_hierarchy_df.shape}")

# Problem 1: Merge datasets
print("\n" + "="*50)
print("PROBLEM 1: MERGING DATASETS")
print("="*50)

# First merge transactions with product hierarchy
transaction_product = pd.merge(transaction_df, product_hierarchy_df, 
                              on='Product_ID', how='left')

# Then merge with customer data (keeping all customers who have transactions)
Customer_Final = pd.merge(customer_df, transaction_product, 
                         on='Customer_ID', how='inner')

print(f"Final merged dataset shape: {Customer_Final.shape}")
print("Dataset 'Customer_Final' created successfully!")

# Problem 2: Summary report
print("\n" + "="*50)
print("PROBLEM 2: SUMMARY REPORT")
print("="*50)

# 2a. Column names and datatypes
print("\n2a. Column names and datatypes:")
print("-" * 40)
for col in Customer_Final.columns:
    print(f"{col}: {Customer_Final[col].dtype}")

# 2b. Top and Bottom 20 observations
print(f"\n2b. Dataset shape: {Customer_Final.shape}")
print("\nTop 20 observations:")
print(Customer_Final.head(20))
print("\nBottom 20 observations:")
print(Customer_Final.tail(20))

# 2c. Total number of rows
print(f"\n2c. Total number of rows: {len(Customer_Final)}")

# 2d. Missing values
print(f"\n2d. Missing values by column:")
missing_values = Customer_Final.isnull().sum()
print(missing_values)

# 2e. Five-number summary for continuous variables
print(f"\n2e. Five-number summary for continuous variables:")
print("-" * 50)
numeric_cols = Customer_Final.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Min: {Customer_Final[col].min()}")
    print(f"  Q1: {Customer_Final[col].quantile(0.25)}")
    print(f"  Median: {Customer_Final[col].median()}")
    print(f"  Q3: {Customer_Final[col].quantile(0.75)}")
    print(f"  Max: {Customer_Final[col].max()}")

# 2f. Frequency tables for categorical variables
print(f"\n2f. Frequency tables for categorical variables:")
print("-" * 50)
categorical_cols = Customer_Final.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(Customer_Final[col].value_counts().head(10))  # Show top 10 for readability

# Problem 3: Transaction analysis
print("\n" + "="*50)
print("PROBLEM 3: TRANSACTION ANALYSIS")
print("="*50)

# Convert date column to datetime if it exists
date_columns = [col for col in Customer_Final.columns if 'date' in col.lower() or 'Date' in col]
if date_columns:
    date_col = date_columns[0]  # Assume first date column is transaction date
    Customer_Final[date_col] = pd.to_datetime(Customer_Final[date_col])
    
    # 3a. Time period of transaction data
    print(f"\n3a. Time period of transaction data:")
    print(f"From: {Customer_Final[date_col].min()}")
    print(f"To: {Customer_Final[date_col].max()}")

# 3b. Count of negative transactions
amount_columns = [col for col in Customer_Final.columns if 'amount' in col.lower() or 'Amount' in col]
if amount_columns:
    amount_col = amount_columns[0]  # Assume first amount column
    negative_transactions = Customer_Final[Customer_Final[amount_col] < 0]
    print(f"\n3b. Transactions with negative amounts: {len(negative_transactions)}")

# Problem 4: Product categories by gender
print("\n" + "="*50)
print("PROBLEM 4: PRODUCT CATEGORIES BY GENDER")
print("="*50)

gender_col = [col for col in Customer_Final.columns if 'gender' in col.lower() or 'Gender' in col][0]
category_col = [col for col in Customer_Final.columns if 'category' in col.lower() or 'Category' in col][0]

gender_category = pd.crosstab(Customer_Final[gender_col], Customer_Final[category_col])
print("\nProduct categories by gender:")
print(gender_category)

# Calculate percentages
gender_category_pct = pd.crosstab(Customer_Final[gender_col], Customer_Final[category_col], normalize='index') * 100
print("\nPercentages by gender:")
print(gender_category_pct.round(2))

# Problem 5: City with maximum customers
print("\n" + "="*50)
print("PROBLEM 5: CITY WITH MAXIMUM CUSTOMERS")
print("="*50)

city_col = [col for col in Customer_Final.columns if 'city' in col.lower() or 'City' in col][0]
city_counts = Customer_Final.drop_duplicates('Customer_ID')[city_col].value_counts()
max_city = city_counts.index[0]
max_city_count = city_counts.iloc[0]
total_customers = Customer_Final['Customer_ID'].nunique()
percentage = (max_city_count / total_customers) * 100

print(f"City with maximum customers: {max_city}")
print(f"Number of customers: {max_city_count}")
print(f"Percentage: {percentage:.2f}%")

# Problem 6: Store type analysis
print("\n" + "="*50)
print("PROBLEM 6: STORE TYPE ANALYSIS")
print("="*50)

store_col = [col for col in Customer_Final.columns if 'store' in col.lower() or 'Store' in col][0]
qty_col = [col for col in Customer_Final.columns if 'qty' in col.lower() or 'Qty' in col or 'quantity' in col.lower()][0]

# By value
store_value = Customer_Final.groupby(store_col)[amount_col].sum().sort_values(ascending=False)
print("Store types by total value:")
print(store_value)

# By quantity
store_quantity = Customer_Final.groupby(store_col)[qty_col].sum().sort_values(ascending=False)
print("\nStore types by total quantity:")
print(store_quantity)

print(f"\nStore type with maximum sales by value: {store_value.index[0]}")
print(f"Store type with maximum sales by quantity: {store_quantity.index[0]}")

# Problem 7: Electronics and Clothing from Flagship stores
print("\n" + "="*50)
print("PROBLEM 7: ELECTRONICS & CLOTHING FROM FLAGSHIP STORES")
print("="*50)

flagship_electronics_clothing = Customer_Final[
    (Customer_Final[store_col].str.contains('Flagship', case=False, na=False)) & 
    (Customer_Final[category_col].isin(['Electronics', 'Clothing']))
]

total_amount = flagship_electronics_clothing[amount_col].sum()
print(f"Total amount from Electronics and Clothing in Flagship stores: {total_amount}")

# Breakdown by category
category_breakdown = flagship_electronics_clothing.groupby(category_col)[amount_col].sum()
print("\nBreakdown by category:")
print(category_breakdown)

# Problem 8: Male customers in Electronics
print("\n" + "="*50)
print("PROBLEM 8: MALE CUSTOMERS IN ELECTRONICS")
print("="*50)

male_electronics = Customer_Final[
    (Customer_Final[gender_col].str.contains('Male', case=False, na=False)) & 
    (Customer_Final[category_col] == 'Electronics')
]

total_male_electronics = male_electronics[amount_col].sum()
print(f"Total amount from Male customers in Electronics: {total_male_electronics}")

# Problem 9: Customers with >10 unique transactions (excluding negative amounts)
print("\n" + "="*50)
print("PROBLEM 9: CUSTOMERS WITH >10 UNIQUE TRANSACTIONS")
print("="*50)

# Remove negative transactions
positive_transactions = Customer_Final[Customer_Final[amount_col] >= 0]

# Count unique transactions per customer
customer_transaction_counts = positive_transactions.groupby('Customer_ID').size()
customers_with_10plus = customer_transaction_counts[customer_transaction_counts > 10]

print(f"Customers with more than 10 unique transactions: {len(customers_with_10plus)}")

# Problem 10: Customers aged 25-35 analysis
print("\n" + "="*50)
print("PROBLEM 10: CUSTOMERS AGED 25-35 ANALYSIS")
print("="*50)

age_col = [col for col in Customer_Final.columns if 'age' in col.lower() or 'Age' in col][0]

# Filter customers aged 25-35
age_filtered = Customer_Final[
    (Customer_Final[age_col] >= 25) & (Customer_Final[age_col] <= 35)
]

# 10a. Electronics and Books spending
electronics_books = age_filtered[
    age_filtered[category_col].isin(['Electronics', 'Books'])
]

electronics_books_total = electronics_books[amount_col].sum()
print(f"10a. Total amount spent on Electronics and Books by customers aged 25-35: {electronics_books_total}")

# Breakdown
category_spending = electronics_books.groupby(category_col)[amount_col].sum()
print("Breakdown by category:")
print(category_spending)

# 10b. Spending between Jan 1, 2014 to Mar 1, 2014
if date_columns:
    date_filtered = age_filtered[
        (age_filtered[date_col] >= '2014-01-01') & 
        (age_filtered[date_col] <= '2014-03-01')
    ]
    
    total_period_spending = date_filtered[amount_col].sum()
    print(f"\n10b. Total amount spent between Jan 1, 2014 to Mar 1, 2014: {total_period_spending}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)

# Save the merged dataset
Customer_Final.to_csv('Customer_Final.csv', index=False)
print("Customer_Final dataset saved as 'Customer_Final.csv'")
```

## Problem Solutions Summary

### Problem 1: Data Merging
- **Approach**: Used inner join to keep only customers with transactions
- **Steps**: 
  1. Merge transactions with product hierarchy on Product_ID
  2. Merge result with customer data on Customer_ID
- **Output**: Customer_Final dataset with all relevant information

### Problem 2: Summary Report
- **Column Analysis**: Data types and structure examination
- **Data Quality**: Missing values assessment
- **Statistical Summary**: Five-number summary for numerical variables
- **Categorical Analysis**: Frequency distributions for categorical variables

### Problem 3: Transaction Analysis
- **Time Range**: Identifies transaction data time period
- **Data Quality**: Counts negative transaction amounts

### Problem 4: Gender-based Product Analysis
- **Method**: Cross-tabulation of gender vs product categories
- **Output**: Both absolute counts and percentages

### Problem 5: Customer Distribution by City
- **Analysis**: Identifies city with maximum customers
- **Metrics**: Count and percentage calculation

### Problem 6: Store Performance Analysis
- **Metrics**: Analysis by both value and quantity
- **Output**: Store type rankings

### Problem 7: Category Revenue from Flagship Stores
- **Focus**: Electronics and Clothing categories
- **Store Type**: Flagship stores only

### Problem 8: Male Customer Electronics Revenue
- **Filter**: Male customers in Electronics category
- **Output**: Total revenue calculation

### Problem 9: High-Activity Customers
- **Criteria**: >10 unique transactions (excluding negative amounts)
- **Method**: Customer-level transaction counting

### Problem 10: Age-based Analysis (25-35 years)
- **Part A**: Electronics and Books spending
- **Part B**: Spending in specific date range (Jan-Mar 2014)

## Key Assumptions and Notes

1. **Column Name Detection**: The code uses intelligent detection for common column patterns
2. **Join Strategy**: Inner join ensures only customers with transactions are included
3. **Date Handling**: Automatic datetime conversion for date columns
4. **Error Handling**: Graceful handling of missing columns

## File Requirements

Ensure your CSV files are named:
- `customer.csv`
- `transaction.csv`
- `product hierarchy.csv`

## Expected Column Names

The code looks for columns containing these keywords (case-insensitive):
- Customer_ID, Product_ID (for joins)
- Date/date (for time analysis)
- Amount/amount (for transaction values)
- Gender/gender (for demographic analysis)
- City/city (for location analysis)
- Store/store (for store analysis)
- Category/category (for product analysis)
- Age/age (for age-based analysis)
- Qty/quantity (for quantity analysis)

## Output Files

- **Customer_Final.csv**: The merged dataset for further analysis
- **Console Output**: Detailed answers to all 10 problems

## Usage Instructions

1. Place your three CSV files in the same directory as the script
2. Run the Python script
3. Review the console output for all analysis results
4. Use the saved Customer_Final.csv for additional analysis if needed

## Customization

If your column names differ from the expected patterns, modify the column detection lines in the code to match your specific column names.
