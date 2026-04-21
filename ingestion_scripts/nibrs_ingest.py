import pandas as pd
import os
import glob
import numpy as np


BASE_DIR = "nibrs_years"
CROSSWALK_FILE = "35158-0001-Data.tsv"
OUTPUT_FILE = "master_norm_crime_2017_2023.csv"

print("Starting Reporting-Adjusted NIBRS Ingestion...")


crosswalk = pd.read_csv(CROSSWALK_FILE, sep='\t', low_memory=False,
                        usecols=['ORI9', 'ADDRESS_ZIP'],
                        dtype={'ORI9': str, 'ADDRESS_ZIP': str})


crosswalk = crosswalk.drop_duplicates(subset=['ORI9'])

all_years_summary = []


target_years = sorted([d for d in os.listdir(BASE_DIR) if d.isdigit()])

for year in target_years:
    year_path = os.path.join(BASE_DIR, year)
    

    search_pattern = os.path.join(year_path, "*-0002-Data.tsv")
    matching_files = glob.glob(search_pattern)
    

    if not matching_files:
        print(f"   [!] Skipping {year}: No match for *-0002-Data.tsv")

        if os.path.exists(year_path):
            print(f"       Found in folder: {os.listdir(year_path)}") 
        continue
    

    data_file = matching_files[0]
    print(f"   [+] Processing {year}: {os.path.basename(data_file)}")
    

    df = pd.read_csv(data_file, sep='\t', low_memory=False,
                     usecols=['V1003', 'V1004'],
                     dtype={'V1003': str, 'V1004': str})
    

    df = df.merge(crosswalk, left_on='V1003', right_on='ORI9', how='left')
    
    zip_stats = df.groupby('ADDRESS_ZIP').agg(
        raw_incident_count=('V1004', 'nunique'),
        active_agencies=('V1003', 'nunique')
    ).reset_index()
    

    zip_stats['incidents_per_agency'] = zip_stats['raw_incident_count'] / zip_stats['active_agencies']
    
    zip_stats['YEAR'] = int(year)
    all_years_summary.append(zip_stats)


if all_years_summary:
    master_df = pd.concat(all_years_summary, ignore_index=True)
    

    master_df = master_df.rename(columns={'ADDRESS_ZIP': 'ZIP'})
    

    master_df = master_df.dropna(subset=['ZIP'])
    

    master_df['ZIP'] = master_df['ZIP'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    

    master_df = master_df[(master_df['ZIP'].str.len() == 5) & (master_df['ZIP'] != '00000')]
    
    zip_counts = master_df['ZIP'].value_counts()
    valid_zips = zip_counts[zip_counts >= 3].index
    master_df = master_df[master_df['ZIP'].isin(valid_zips)]
    

    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[!] SUCCESS: Created {OUTPUT_FILE}")
    print(f"Total clean entries: {len(master_df)}")
    print("\nSample of Normalized & Cleaned Data:")
    print(master_df.sort_values(['ZIP', 'YEAR']).head(10))
else:
    print("\n[!] Processing failed. No data collected.")