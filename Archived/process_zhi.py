import pandas as pd

file_path = "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
df_zillow = pd.read_csv(file_path)

geo_columns = ['RegionName', 'City', 'State', 'Metro', 'CountyName']

date_columns = [col for col in df_zillow.columns if col.startswith('20')]

columns_to_keep = geo_columns + date_columns
df_clean = df_zillow[columns_to_keep].copy()

df_clean = df_clean.rename(columns={'RegionName': 'ZIP_Code'})

df_clean['ZIP_Code'] = df_clean['ZIP_Code'].astype(str)

print(f"Dataframe shape: {df_clean.shape}")
print(df_clean.head())