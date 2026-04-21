import pandas as pd
import geopandas as gpd

# 1. Load the JSON file into a Pandas DataFrame
# (Replace 'us_cities.json' with your actual file name)
df_cities = pd.read_json('cities.json')

# 2. Extract only the columns you need for the Spatial Join
core_columns = ['city', 'state', 'latitude', 'longitude']
df_cities_clean = df_cities[core_columns].copy()

# 3. Convert the standard Pandas DataFrame into a GeoDataFrame
# This translates the text coordinates into actual mathematical points on a map
gdf_city_nodes = gpd.GeoDataFrame(
    df_cities_clean, 
    geometry=gpd.points_from_xy(df_cities_clean.longitude, df_cities_clean.latitude),
    crs="EPSG:4326" # Standard GPS coordinate projection
)

print(f"Successfully loaded {len(gdf_city_nodes)} city nodes.")
print(gdf_city_nodes.head())