import pandas as pd

# ==========================================
# 1. Load Data Safely 
# ==========================================
# Force postal_code to be a string to preserve leading zeros (e.g., '06706')
file_path = 'RDC_Inventory_Core_Metrics_Zip_History.csv'
df_realtor = pd.read_csv(file_path, dtype={'postal_code': str})

# ==========================================
# 2. Surgical Feature Extraction (Keeping History)
# ==========================================
# We now KEEP the 'month_date_yyyymm' column, but still drop the _mm and _yy bloat.
core_columns = [
    'month_date_yyyymm',             # Feature: Temporal Identifier (CRITICAL NOW)
    'postal_code',
    'zip_name',
    'median_listing_price',          # Target Variable
    'active_listing_count',          # Feature: Market Supply
    'median_days_on_market',         # Feature: Market Velocity / Demand
    'new_listing_count',             # Feature: Market Momentum
    'price_reduced_share',           # Feature: Cooling Market Indicator
    'median_square_feet',            # Feature: Baseline Property Attribute
    'median_listing_price_per_square_foot' 
]

# We are no longer filtering by month, we take the whole dataset!
df_clean = df_realtor[core_columns].copy()

# ==========================================
# 3. Rename Columns for Consistency
# ==========================================
df_clean = df_clean.rename(columns={
    'month_date_yyyymm': 'YearMonth', # Renamed for clarity
    'postal_code': 'ZIP_Code',
    'median_listing_price': 'Actual_Median_Price',
    'active_listing_count': 'Active_Listings',
    'median_days_on_market': 'Days_On_Market',
    'price_reduced_share': 'Price_Reduced_Share',
    'median_square_feet': 'Median_SqFt',
    'median_listing_price_per_square_foot': 'Price_Per_SqFt'
})

# ==========================================
# 4. Convert Date for Time-Series Analysis
# ==========================================
# Realtor.com stores dates as integers like 202603. 
# Converting this to a true Pandas Datetime object makes charting and math much easier.
df_clean['YearMonth'] = pd.to_datetime(df_clean['YearMonth'].astype(str), format='%Y%m')

# ==========================================
# 5. Missing Value Imputation
# ==========================================
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# Sort chronologically so your historical data is in order
df_clean = df_clean.sort_values(by=['ZIP_Code', 'YearMonth']).reset_index(drop=True)

print(f"Historical DataFrame Shape: {df_clean.shape}")
print(df_clean.head())