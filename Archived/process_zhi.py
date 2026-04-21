import pandas as pd

# 1. Load the Zillow CSV
file_path = "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
df_zillow = pd.read_csv(file_path)

# 2. Define the core geographic columns you want
geo_columns = ['RegionName', 'City', 'State', 'Metro', 'CountyName']

# 3. Automatically grab EVERY date column (they all start with '20')
date_columns = [col for col in df_zillow.columns if col.startswith('20')]

# 4. Combine the lists and filter the DataFrame
columns_to_keep = geo_columns + date_columns
df_clean = df_zillow[columns_to_keep].copy()

# 5. Rename RegionName to ZIP_Code to prevent merging headaches later
df_clean = df_clean.rename(columns={'RegionName': 'ZIP_Code'})

# 6. Convert ZIP Codes to strings
df_clean['ZIP_Code'] = df_clean['ZIP_Code'].astype(str)

# Show the shape (rows, columns) to verify you captured all months
print(f"Dataframe shape: {df_clean.shape}")
print(df_clean.head())