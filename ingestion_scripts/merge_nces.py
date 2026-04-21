import pandas as pd
import numpy as np
import os


print("Initializing Master Merge (Final Infrastructure Edition)...")
crime_pop = pd.read_csv('final_training_features_2017_2026.csv', dtype={'ZIP': str})
income_df = pd.read_csv('master_zip_income_2017_2026.csv', dtype={'ZIP': str})
zvhi_df = pd.read_csv('master_zvhi_yearly_2017_2026.csv', dtype={'ZIP': str})
schools_df = pd.read_csv('master_schools_2017_2025.csv', dtype={'ZIP': str})


print("   [+] Calculating Weather Stability from quarterly snapshots...")

weather_df = pd.read_csv('top_1000_cities_monthly_weather.csv', dtype={'ZIP_Code': str})


weather_df.columns = weather_df.columns.str.strip() 


weather_df['ZIP'] = weather_df['ZIP_Code'].str.zfill(5)
weather_df['YEAR'] = pd.to_datetime(weather_df['YearMonth']).dt.year


weather_stats = weather_df.groupby(['ZIP', 'YEAR'])['TAVG'].agg(['mean', 'std']).reset_index()
weather_stats.rename(columns={
    'mean': 'yearly_temp_avg', 
    'std': 'temp_stability_score'
}, inplace=True)


print("   [+] Aggregating raw school counts...")
def convert_school_year(yr):
    yr_str = str(yr)
    if len(yr_str) == 4: return 2000 + int(yr_str[2:])
    return yr

schools_df['YEAR'] = schools_df['SCHOOL_YEAR'].apply(convert_school_year)
school_counts = schools_df.groupby(['ZIP', 'YEAR']).size().reset_index(name='school_count')
school_counts['ZIP'] = school_counts['ZIP'].str.zfill(5)


print("   [+] Building final training matrix...")
master = crime_pop.merge(income_df, on=['ZIP', 'YEAR'], how='left')
master = master.merge(school_counts, on=['ZIP', 'YEAR'], how='left')
master = master.merge(weather_stats, on=['ZIP', 'YEAR'], how='left')
master = master.merge(zvhi_df, on=['ZIP', 'YEAR', 'state_id'], how='left')


print("   [+] Engineering Infrastructure & Safety metrics...")

master = master.sort_values(['ZIP', 'YEAR'])


master['schools_per_capita'] = (master['school_count'] / master['population_estimate']) * 1000


for col in ['school_count', 'schools_per_capita', 'yearly_temp_avg', 'temp_stability_score']:
    master[col] = master.groupby('ZIP')[col].ffill().bfill()
    state_med = master.groupby(['state_id', 'YEAR'])[col].transform('median')
    master[col] = master[col].fillna(state_med)


crime_pop['crime_rolling'] = crime_pop.groupby('ZIP')['final_crime'].transform(lambda x: x.rolling(3, min_periods=1).mean())
master['crime_capita_rolling'] = (master.groupby('ZIP')['crime_rolling'].ffill().bfill() / master['population_estimate'])


master['price_momentum'] = master.groupby('ZIP')['ZVHI'].pct_change().fillna(0)


master = master[~master['state_id'].isin(['PR', 'GU', 'VI', 'AS', 'MP'])]
master = master.dropna(subset=['ZVHI'])

final_cols = [
    'ZIP', 'YEAR', 'state_id', 'population_estimate', 'ZVHI', 'median_income',
    'yearly_temp_avg', 'temp_stability_score', 'crime_capita_rolling', 
    'school_count', 'schools_per_capita', 'price_momentum'
]

master[final_cols].to_csv('XGBOOST_FAIR_VALUE_READY.csv', index=False)
print(f"\n SUCCESS: Processed {len(master)} samples.")
print(f"Features: {final_cols[3:]}")