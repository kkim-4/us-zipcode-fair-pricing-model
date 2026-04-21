import pandas as pd


file_path = 'RDC_Inventory_Core_Metrics_Zip_History.csv'
df_realtor = pd.read_csv(file_path, dtype={'postal_code': str})

core_columns = [
    'month_date_yyyymm',            
    'postal_code',
    'zip_name',
    'median_listing_price',       
    'active_listing_count',        
    'median_days_on_market',         
    'new_listing_count',            
    'price_reduced_share',           
    'median_square_feet',            
    'median_listing_price_per_square_foot' 
]

df_clean = df_realtor[core_columns].copy()


df_clean = df_clean.rename(columns={
    'month_date_yyyymm': 'YearMonth',
    'postal_code': 'ZIP_Code',
    'median_listing_price': 'Actual_Median_Price',
    'active_listing_count': 'Active_Listings',
    'median_days_on_market': 'Days_On_Market',
    'price_reduced_share': 'Price_Reduced_Share',
    'median_square_feet': 'Median_SqFt',
    'median_listing_price_per_square_foot': 'Price_Per_SqFt'
})

df_clean['YearMonth'] = pd.to_datetime(df_clean['YearMonth'].astype(str), format='%Y%m')

numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

df_clean = df_clean.sort_values(by=['ZIP_Code', 'YearMonth']).reset_index(drop=True)

print(f"Historical DataFrame Shape: {df_clean.shape}")
print(df_clean.head())