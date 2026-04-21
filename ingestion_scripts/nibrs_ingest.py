import pandas as pd
import os
import glob
import numpy as np

# 1. SETUP PATHS
BASE_DIR = "nibrs_years"
CROSSWALK_FILE = "35158-0001-Data.tsv"
OUTPUT_FILE = "master_norm_crime_2017_2023.csv"

print("Starting Reporting-Adjusted NIBRS Ingestion...")

# 2. LOAD CROSSWALK
# We use ORI9 to match V1003, and ADDRESS_ZIP for the mapping
crosswalk = pd.read_csv(CROSSWALK_FILE, sep='\t', low_memory=False,
                        usecols=['ORI9', 'ADDRESS_ZIP'],
                        dtype={'ORI9': str, 'ADDRESS_ZIP': str})

# Ensure we have a clean 1-to-1 mapping for the bridge
crosswalk = crosswalk.drop_duplicates(subset=['ORI9'])

all_years_summary = []

# 3. THE NORMALIZING CRAWLER
# List of years 2017-2023
target_years = sorted([d for d in os.listdir(BASE_DIR) if d.isdigit()])

for year in target_years:
    year_path = os.path.join(BASE_DIR, year)
    
    # 1. Use glob to expand the wildcard *
    search_pattern = os.path.join(year_path, "*-0002-Data.tsv")
    matching_files = glob.glob(search_pattern)
    
    # 2. Check if glob actually found anything
    if not matching_files:
        print(f"   [!] Skipping {year}: No match for *-0002-Data.tsv")
        # Debugging help to see what is actually in the directory
        if os.path.exists(year_path):
            print(f"       Found in folder: {os.listdir(year_path)}") 
        continue
    
    # 3. Take the first file found
    data_file = matching_files[0]
    print(f"   [+] Processing {year}: {os.path.basename(data_file)}")
    
    # Load DS0002 (Administrative Segment)
    # V1003 = ORI (Agency), V1004 = Incident ID
    df = pd.read_csv(data_file, sep='\t', low_memory=False,
                     usecols=['V1003', 'V1004'],
                     dtype={'V1003': str, 'V1004': str})
    
    # Merge with Crosswalk to get ZIPs
    df = df.merge(crosswalk, left_on='V1003', right_on='ORI9', how='left')
    
    # NORMALIZATION LOGIC:
    # We group by ZIP to see how many incidents happened AND how many agencies reported them
    zip_stats = df.groupby('ADDRESS_ZIP').agg(
        raw_incident_count=('V1004', 'nunique'),
        active_agencies=('V1003', 'nunique')
    ).reset_index()
    
    # Calculate the "Normalized Crime Load" 
    # This prevents the 2023 'Reporting Jump' from skewing your model
    zip_stats['incidents_per_agency'] = zip_stats['raw_incident_count'] / zip_stats['active_agencies']
    
    zip_stats['YEAR'] = int(year)
    all_years_summary.append(zip_stats)

# 4. CONSOLIDATE & FILTER
if all_years_summary:
    master_df = pd.concat(all_years_summary, ignore_index=True)
    
    # Clean up column names and formatting
    master_df = master_df.rename(columns={'ADDRESS_ZIP': 'ZIP'})
    
    # Drop rows where ZIP is missing before string manipulation
    master_df = master_df.dropna(subset=['ZIP'])
    
    # Ensure 5-digit string, stripping any accidental ".0" decimals from pandas
    master_df['ZIP'] = master_df['ZIP'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    
    # FILTER 1: Remove junk ZIPs and the '00000' unmapped bucket
    master_df = master_df[(master_df['ZIP'].str.len() == 5) & (master_df['ZIP'] != '00000')]
    
    # FILTER 2: Time-Series Continuity
    # Keep only ZIP codes that have reported data for at least 3 years.
    # This prevents the XGBoost model from learning on "one-hit wonder" agencies.
    zip_counts = master_df['ZIP'].value_counts()
    valid_zips = zip_counts[zip_counts >= 3].index
    master_df = master_df[master_df['ZIP'].isin(valid_zips)]
    
    # Save the feature set
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[!] SUCCESS: Created {OUTPUT_FILE}")
    print(f"Total clean entries: {len(master_df)}")
    print("\nSample of Normalized & Cleaned Data:")
    print(master_df.sort_values(['ZIP', 'YEAR']).head(10))
else:
    print("\n[!] Processing failed. No data collected.")