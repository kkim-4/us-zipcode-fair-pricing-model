import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("Building 50-State Skeleton (2017-2026)...")

universe = pd.read_csv('uszips.csv', dtype={'zip': str, 'county_fips': str})


non_states = ['PR', 'GU', 'VI', 'AS', 'MP']
universe = universe[~universe['state_id'].isin(non_states)].copy()
universe['ZIP'] = universe['zip'].str.zfill(5)


years = pd.DataFrame({'YEAR': range(2017, 2027)})
skeleton = universe[['ZIP', 'county_fips', 'state_id']].assign(key=1).merge(years.assign(key=1), on='key').drop('key', axis=1)


print("Aligning Crime, Population, and Crosswalk data...")
crime_df = pd.read_csv('master_norm_crime_2017_2023.csv', dtype={'ZIP': str})
crime_df['ZIP'] = crime_df['ZIP'].str.zfill(5)

pop_df = pd.read_csv('master_zip_population_2017_2026.csv', dtype={'ZIP': str})
pop_df['ZIP'] = pop_df['ZIP'].str.zfill(5)


crime_with_geo = crime_df.merge(universe[['ZIP', 'county_fips']], on='ZIP', how='left')


county_baselines = crime_with_geo.groupby(['county_fips', 'YEAR'])['incidents_per_agency'].mean().reset_index()
county_baselines = county_baselines.rename(columns={'incidents_per_agency': 'county_avg'})

master = skeleton.merge(pop_df, on=['ZIP', 'YEAR'], how='left')
master = master.merge(crime_df, on=['ZIP', 'YEAR'], how='left')
master = master.merge(county_baselines, on=['county_fips', 'YEAR'], how='left')

master['crime_base'] = master['incidents_per_agency'].fillna(master['county_avg'])


print("Projecting 2024-2026 using Log-Linear trends...")

def project_non_negative(group):
  
    known = group[group['YEAR'] <= 2023].dropna(subset=['crime_base'])
    
    if len(known) < 2:
       
        return group.assign(final_crime=group['crime_base'].fillna(master['crime_base'].median()))

    X = known[['YEAR']].values
    
    y = np.log1p(known['crime_base'].values)
    
    model = LinearRegression().fit(X, y)
    
    
    all_years = group[['YEAR']].values
   
    preds = np.expm1(model.predict(all_years))
    
    group['final_crime'] = group['crime_base'].fillna(pd.Series(preds, index=group.index))
    
    group['final_crime'] = group['final_crime'].clip(lower=0)
    return group


final_matrix = master.groupby('ZIP', group_keys=False).apply(project_non_negative)


state_pop_median = final_matrix.groupby(['state_id', 'YEAR'])['population_estimate'].transform('median')
final_matrix['population_estimate'] = final_matrix['population_estimate'].fillna(state_pop_median)

final_output = final_matrix[['ZIP', 'YEAR', 'state_id', 'population_estimate', 'final_crime']]
final_output.to_csv('final_training_features_2017_2026.csv', index=False)

print("\n[!] SUCCESS: Created 'final_training_features_2017_2026.csv'")
print(final_output[final_output['ZIP'].isin(['64844', '64849'])].sort_values(['ZIP', 'YEAR']))