import pandas as pd
import geopandas as gpd

df_cities = pd.read_json('cities.json')

core_columns = ['city', 'state', 'latitude', 'longitude']
df_cities_clean = df_cities[core_columns].copy()


gdf_city_nodes = gpd.GeoDataFrame(
    df_cities_clean, 
    geometry=gpd.points_from_xy(df_cities_clean.longitude, df_cities_clean.latitude),
    crs="EPSG:4326" 
)

print(f"Successfully loaded {len(gdf_city_nodes)} city nodes.")
print(gdf_city_nodes.head())