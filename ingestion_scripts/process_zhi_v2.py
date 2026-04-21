import pandas as pd


zillow_file = 'Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
zvhi_raw = pd.read_csv(zillow_file, dtype={'RegionName': str})


date_cols = [c for c in zvhi_raw.columns if c.startswith(('201', '202'))]


print("Melting 100+ monthly columns into a single time-series...")
zvhi_long = zvhi_raw.melt(
    id_vars=['RegionName', 'State'], 
    value_vars=date_cols, 
    var_name='Date', 
    value_name='ZVHI'
)

zvhi_long['YEAR'] = pd.to_datetime(zvhi_long['Date']).dt.year
zvhi_yearly = zvhi_long.groupby(['RegionName', 'YEAR', 'State'])['ZVHI'].mean().reset_index()


zvhi_yearly = zvhi_yearly.rename(columns={'RegionName': 'ZIP', 'State': 'state_id'})
zvhi_yearly['ZIP'] = zvhi_yearly['ZIP'].str.zfill(5)


zvhi_yearly = zvhi_yearly[~zvhi_yearly['state_id'].isin(['PR', 'GU', 'VI', 'AS', 'MP'])]

zvhi_yearly.to_csv('master_zvhi_yearly_2017_2026.csv', index=False)
print("Created master_zvhi_yearly_2017_2026.csv")