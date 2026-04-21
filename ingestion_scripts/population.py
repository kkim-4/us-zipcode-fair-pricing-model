import pandas as pd
import numpy as np
import sys

def load_and_clean(file_path):
    print(f"Reading {file_path}...")
    # Skip the metadata row (row 1)
    df = pd.read_csv(file_path, skiprows=[1])
    
    # 1. FIND THE POPULATION COLUMN
    # Census 2020 uses P1_001N, 2010 often uses P0010001 or P1_001N
    possible_pop_cols = ['P1_001N', 'P001001', 'P0010001']
    found_col = None
    
    for col in possible_pop_cols:
        if col in df.columns:
            found_col = col
            break
            
    if not found_col:
        print(f"\n[!] ERROR: Could not find population column in {file_path}")
        print(f"Available columns are: {df.columns.tolist()[:10]}...") 
        sys.exit()

    # 2. RENAME AND CLEAN
    df = df.rename(columns={found_col: 'pop'})
    
    # Extract 5-digit ZIP from GEO_ID (e.g., 860Z200US00601 -> 00601)
    # Some files use 'GEO_ID', others might use 'GEOID'
    geo_col = 'GEO_ID' if 'GEO_ID' in df.columns else 'GEOID'
    df['ZIP'] = df[geo_col].astype(str).str.slice(-5)
    
    # Convert pop to numeric
    df['pop'] = pd.to_numeric(df['pop'], errors='coerce').fillna(0).astype(int)
    
    return df[['ZIP', 'pop']]

# --- MAIN EXECUTION ---
try:
    pop_2010 = load_and_clean('DECENNIALSF12010.P1-Data.csv')
    pop_2020 = load_and_clean('DECENNIALDHC2020.P1-Data.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")
    sys.exit()

# Merge on ZIP
master_pop = pd.merge(pop_2010, pop_2020, on='ZIP', suffixes=('_2010', '_2020'))

# Calculate the annual growth rate (slope)
master_pop['growth_rate'] = (master_pop['pop_2020'] - master_pop['pop_2010']) / 10

# Generate columns for 2017 through 2026
target_years = range(2017, 2027)
print(f"Interpolating years {min(target_years)} to {max(target_years)}...")

for year in target_years:
    years_elapsed = year - 2010
    master_pop[f'pop_{year}'] = (
        master_pop['pop_2010'] + (master_pop['growth_rate'] * years_elapsed)
    ).astype(int)

# Transform to 'Long Format'
long_format_cols = ['ZIP'] + [f'pop_{y}' for y in target_years]
pop_long = master_pop[long_format_cols].melt(
    id_vars='ZIP', 
    var_name='YEAR', 
    value_name='population_estimate'
)

pop_long['YEAR'] = pop_long['YEAR'].str.replace('pop_', '').astype(int)

# Save the master lookup table
pop_long.to_csv('master_zip_population_2017_2026.csv', index=False)

print("\n[!] SUCCESS: Created 'master_zip_population_2017_2026.csv'")
print(pop_long.head(10))