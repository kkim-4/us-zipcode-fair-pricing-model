import pandas as pd

# 1. LOAD REALTOR DATA
realtor_file = 'RDC_Inventory_Core_Metrics_Zip_History.csv'
realtor_df = pd.read_csv(realtor_file, dtype={'postal_code': str})

# 2. EXTRACT YEAR AND AGGREGATE
# Format: 202603 -> 2026
realtor_df['YEAR'] = (realtor_df['month_date_yyyymm'] // 100)
realtor_df = realtor_df.rename(columns={'postal_code': 'ZIP'})
realtor_df['ZIP'] = realtor_df['ZIP'].str.zfill(5)

# Calculate yearly average listing price
realtor_yearly = realtor_df.groupby(['ZIP', 'YEAR'])['median_listing_price'].mean().reset_index()

realtor_yearly.to_csv('master_realtor_yearly.csv', index=False)
print("Created master_realtor_yearly.csv")
print(realtor_yearly.head())

#