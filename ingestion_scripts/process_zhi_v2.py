import pandas as pd

# 1. LOAD DATA
# RegionName is your ZIP code column
zillow_file = 'Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
zvhi_raw = pd.read_csv(zillow_file, dtype={'RegionName': str})

# 2. IDENTIFY DATE COLUMNS
# We only care about the years in your study (2017-2026)
date_cols = [c for c in zvhi_raw.columns if c.startswith(('201', '202'))]

# 3. MELT FROM WIDE TO LONG
print("Melting 100+ monthly columns into a single time-series...")
zvhi_long = zvhi_raw.melt(
    id_vars=['RegionName', 'State'], 
    value_vars=date_cols, 
    var_name='Date', 
    value_name='ZVHI'
)

# 4. AGGREGATE TO YEARLY
zvhi_long['YEAR'] = pd.to_datetime(zvhi_long['Date']).dt.year
zvhi_yearly = zvhi_long.groupby(['RegionName', 'YEAR', 'State'])['ZVHI'].mean().reset_index()

# 5. CLEAN AND SAVE
zvhi_yearly = zvhi_yearly.rename(columns={'RegionName': 'ZIP', 'State': 'state_id'})
zvhi_yearly['ZIP'] = zvhi_yearly['ZIP'].str.zfill(5)

# Filter for US States only (Focusing on your project scope)
zvhi_yearly = zvhi_yearly[~zvhi_yearly['state_id'].isin(['PR', 'GU', 'VI', 'AS', 'MP'])]

zvhi_yearly.to_csv('master_zvhi_yearly_2017_2026.csv', index=False)
print("Created master_zvhi_yearly_2017_2026.csv")