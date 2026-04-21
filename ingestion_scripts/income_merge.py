import pandas as pd
import glob
import os
import re
import numpy as np


file_pattern = "income_data/*-Data.csv"
files = glob.glob(file_pattern)

income_col = "S1903_C03_001E" 
all_income_data = []

def clean_census_value(val):
    if pd.isna(val) or val == "-" or val == "N" or val == "(X)":
        return np.nan
    str_val = str(val).replace(',', '').replace('+', '').replace('-', '')
    try:
        return float(str_val)
    except ValueError:
        return np.nan


print(f"Found {len(files)} data files. Processing...")

for file in sorted(files):
    file_name = os.path.basename(file)
    
   
    match = re.search(r'20\d{2}', file_name)
    if not match:
        continue
    year = int(match.group())
    

    df = pd.read_csv(file, skiprows=[1], low_memory=False)
    

    df['ZIP'] = df['GEO_ID'].astype(str).str.slice(-5)
    
    if income_col not in df.columns:
        print(f"   [!] Skipping {file_name}: Column {income_col} not found.")
        continue


    df['median_income'] = df[income_col].apply(clean_census_value)
    df['YEAR'] = year
    
    df_clean = df.groupby(['ZIP', 'YEAR'])['median_income'].mean().reset_index()
    
    all_income_data.append(df_clean)
    print(f"   [+] Processed {year}")


if not all_income_data:
    print("No data found! Check your folder path and filenames.")
else:
    master_income = pd.concat(all_income_data, ignore_index=True)

    print("Extrapolating for 2025 and 2026...")
    
    def project_income(group):
        group = group.sort_values('YEAR')
        if len(group) < 2: return group
        
        z = np.polyfit(group['YEAR'], group['median_income'], 1)
        p = np.poly1d(z)
        
        last_year = group['YEAR'].max()
        zip_code = group.iloc[0]['ZIP']
        
        new_rows = []
        for target_year in [2025, 2026]:
            if target_year > last_year:
                projected_val = p(target_year)
                new_rows.append({'ZIP': zip_code, 'YEAR': target_year, 'median_income': round(projected_val, 2)})
        
        return pd.concat([group, pd.DataFrame(new_rows)], ignore_index=True)

    final_income_df = master_income.groupby('ZIP', group_keys=False).apply(project_income)

    final_income_df.to_csv("master_zip_income_2017_2026.csv", index=False)
    print("\n[!] SUCCESS: Created 'master_zip_income_2017_2026.csv'")
    print(final_income_df.head(15))