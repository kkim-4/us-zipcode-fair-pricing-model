import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df_results = pd.read_csv('USA_ZIP_VALUATION_AUDIT_2026.csv', dtype={'ZIP': str})
df_results = df_results.rename(columns={'ZIP': 'ZIP_Code', 'valuation_class': 'Valuation_Status'})
df_results['Valuation_Status'] = df_results['Valuation_Status'].replace('Fairly Priced', 'Fairly Valued')

shapefile_path = "/Users/kevinkim/Downloads/DM_project/tl_2025_us_zcta520.shp"
gdf_zcta = gpd.read_file(shapefile_path)
gdf_zcta = gdf_zcta.rename(columns={'ZCTA5CE20': 'ZIP_Code'})

gdf_conus = gdf_zcta[~gdf_zcta['ZIP_Code'].str.startswith(('00', '96', '99'))].copy()

merged_map = gdf_conus.merge(df_results, on='ZIP_Code', how='left')

merged_map = merged_map.to_crs(epsg=5070)

fig, ax = plt.subplots(1, 1, figsize=(20, 12))

merged_map.plot(ax=ax, color='#eeeeee', linewidth=0)

color_dict = {
    'Undervalued': '#2ecc71',   
    'Fairly Valued': '#bdc3c7', 
    'Overvalued': '#e74c3c'     
}

for status, color in color_dict.items():
    subset = merged_map[merged_map['Valuation_Status'] == status]
    if not subset.empty:
        subset.plot(
            ax=ax, 
            color=color, 
            linewidth=0,
            antialiased=False
        )

ax.set_title('US Housing Market Valuation Audit: Neighborhood Fundamentals vs. Listing Price (2026)', 
             fontsize=22, fontweight='bold', family='serif', pad=20)
ax.set_axis_off()

legend_elements = [mpatches.Patch(facecolor=color, label=status) for status, color in color_dict.items()]
ax.legend(handles=legend_elements, title="Market Status", loc='lower right', fontsize=15, title_fontsize=16)

plt.tight_layout()
plt.savefig('USA_National_Valuation_Audit.png', dpi=300, bbox_inches='tight')
print("National map saved! This might take a minute to render...")
plt.show()