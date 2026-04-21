import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. SETUP & SKELETON
print("Building 50-State Skeleton (2017-2026)...")
# uszips.csv is our spine for the project
universe = pd.read_csv('uszips.csv', dtype={'zip': str, 'county_fips': str})

# Filter: Focus only on the 50 US States + DC
non_states = ['PR', 'GU', 'VI', 'AS', 'MP']
universe = universe[~universe['state_id'].isin(non_states)].copy()
universe['ZIP'] = universe['zip'].str.zfill(5)

# Create 10-year skeleton for every ZIP in the 50 states
years = pd.DataFrame({'YEAR': range(2017, 2027)})
skeleton = universe[['ZIP', 'county_fips', 'state_id']].assign(key=1).merge(years.assign(key=1), on='key').drop('key', axis=1)

# 2. LOAD & ALIGN FEATURES
print("Aligning Crime, Population, and Crosswalk data...")
crime_df = pd.read_csv('master_norm_crime_2017_2023.csv', dtype={'ZIP': str})
crime_df['ZIP'] = crime_df['ZIP'].str.zfill(5)

pop_df = pd.read_csv('master_zip_population_2017_2026.csv', dtype={'ZIP': str})
pop_df['ZIP'] = pop_df['ZIP'].str.zfill(5)

# Merge universe into crime to get County FIPS for the fallback
crime_with_geo = crime_df.merge(universe[['ZIP', 'county_fips']], on='ZIP', how='left')

# 3. COUNTY FALLBACK (Spatial Imputation)
# Calculate average incidents per agency for each county/year
county_baselines = crime_with_geo.groupby(['county_fips', 'YEAR'])['incidents_per_agency'].mean().reset_index()
county_baselines = county_baselines.rename(columns={'incidents_per_agency': 'county_avg'})

# Build the master matrix
master = skeleton.merge(pop_df, on=['ZIP', 'YEAR'], how='left')
master = master.merge(crime_df, on=['ZIP', 'YEAR'], how='left')
master = master.merge(county_baselines, on=['county_fips', 'YEAR'], how='left')

# Use ZIP data if available, otherwise fallback to County average
master['crime_base'] = master['incidents_per_agency'].fillna(master['county_avg'])

# 4. LOG-LINEAR PROJECTION (Fixes the Negative Problem)
print("Projecting 2024-2026 using Log-Linear trends...")

def project_non_negative(group):
    # known = 2017-2023 data
    known = group[group['YEAR'] <= 2023].dropna(subset=['crime_base'])
    
    if len(known) < 2:
        # If no local data, use the global median to keep the XGBoost weights stable
        return group.assign(final_crime=group['crime_base'].fillna(master['crime_base'].median()))

    X = known[['YEAR']].values
    # Log-transform the target to ensure predictions stay positive: ln(y + 1)
    y = np.log1p(known['crime_base'].values)
    
    model = LinearRegression().fit(X, y)
    
    # Predict for all years 2017-2026
    all_years = group[['YEAR']].values
    # Inverse log: exp(y) - 1
    preds = np.expm1(model.predict(all_years))
    
    # Combine actuals (2017-23) with projected trends (2024-26)
    group['final_crime'] = group['crime_base'].fillna(pd.Series(preds, index=group.index))
    # Final safety clip at 0
    group['final_crime'] = group['final_crime'].clip(lower=0)
    return group

# Processing roughly 33,000 ZIP codes
final_matrix = master.groupby('ZIP', group_keys=False).apply(project_non_negative)

# 5. FINAL EXPORT
# Clean up population for missing ZIPs (like 64868) using state medians
state_pop_median = final_matrix.groupby(['state_id', 'YEAR'])['population_estimate'].transform('median')
final_matrix['population_estimate'] = final_matrix['population_estimate'].fillna(state_pop_median)

final_output = final_matrix[['ZIP', 'YEAR', 'state_id', 'population_estimate', 'final_crime']]
final_output.to_csv('final_training_features_2017_2026.csv', index=False)

print("\n[!] SUCCESS: Created 'final_training_features_2017_2026.csv'")
print(final_output[final_output['ZIP'].isin(['64844', '64849'])].sort_values(['ZIP', 'YEAR']))