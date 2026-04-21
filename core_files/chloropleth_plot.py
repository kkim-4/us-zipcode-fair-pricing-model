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

gdf_atlanta = gdf_zcta[gdf_zcta['ZIP_Code'].str.startswith('303')].copy()

merged_map = gdf_atlanta.merge(df_results, on='ZIP_Code', how='left')


color_dict = {
    'Undervalued': '#2ecc71',   
    'Fairly Valued': '#bdc3c7', 
    'Overvalued': '#e74c3c'     
}

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

merged_map.plot(ax=ax, color='#f5f5f5', edgecolor='white', linewidth=0.5)


for status, color in color_dict.items():
    subset = merged_map[merged_map['Valuation_Status'] == status]
    if not subset.empty:
        subset.plot(
            ax=ax, 
            color=color, 
            edgecolor='white', 
            linewidth=0.6,
            zorder=3
        )

ax.set_title('Geospatial Valuation Audit: Atlanta Region (2026)', 
             fontdict={'fontsize': 18, 'fontweight': 'bold', 'family': 'serif'})
ax.set_axis_off() 


legend_elements = [
    mpatches.Patch(facecolor=color, edgecolor='none', label=status)
    for status, color in color_dict.items()
]

ax.legend(handles=legend_elements, 
          title="Model Market Status", 
          loc='lower right', 
          fontsize=12,
          title_fontsize=13,
          frameon=True)


plt.tight_layout()
plt.savefig('atlanta_valuation_map.png', dpi=300, bbox_inches='tight')
print("Map successfully saved as 'atlanta_valuation_map.png'")
plt.show()