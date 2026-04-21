import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. LOAD DATA
df_results = pd.read_csv('USA_ZIP_VALUATION_AUDIT_2026.csv', dtype={'ZIP': str})
df_results = df_results.rename(columns={'ZIP': 'ZIP_Code', 'valuation_class': 'Valuation_Status'})
df_results['Valuation_Status'] = df_results['Valuation_Status'].replace('Fairly Priced', 'Fairly Valued')

# 2. LOAD SHAPEFILE (Optimized)
shapefile_path = "/Users/kevinkim/Downloads/DM_project/tl_2025_us_zcta520.shp"
gdf_zcta = gpd.read_file(shapefile_path)
gdf_zcta = gdf_zcta.rename(columns={'ZCTA5CE20': 'ZIP_Code'})

# 3. FILTER FOR CONTIGUOUS US (Lower 48)
# We exclude ZIPs starting with 967-968 (HI), 995-999 (AK), and 006-009 (PR)
# A simpler way is to use a coordinate box or join with a State shapefile
gdf_conus = gdf_zcta[~gdf_zcta['ZIP_Code'].str.startswith(('00', '96', '99'))].copy()

# 4. MERGE
merged_map = gdf_conus.merge(df_results, on='ZIP_Code', how='left')

# 5. RE-PROJECT (The "IEEE Professional" Look)
# Standard Lat/Long looks "flat". Albers Equal Area (EPSG:5070) gives the US its iconic curve.
merged_map = merged_map.to_crs(epsg=5070)

# 6. PLOT
fig, ax = plt.subplots(1, 1, figsize=(20, 12))

# Background (Grey for missing data/rural areas)
merged_map.plot(ax=ax, color='#eeeeee', linewidth=0)

color_dict = {
    'Undervalued': '#2ecc71',   # Green
    'Fairly Valued': '#bdc3c7', # Grey
    'Overvalued': '#e74c3c'     # Red
}

for status, color in color_dict.items():
    subset = merged_map[merged_map['Valuation_Status'] == status]
    if not subset.empty:
        subset.plot(
            ax=ax, 
            color=color, 
            linewidth=0,  # CRITICAL: No borders at national scale
            antialiased=False
        )

# 7. FORMATTING
ax.set_title('US Housing Market Valuation Audit: Neighborhood Fundamentals vs. Listing Price (2026)', 
             fontsize=22, fontweight='bold', family='serif', pad=20)
ax.set_axis_off()

legend_elements = [mpatches.Patch(facecolor=color, label=status) for status, color in color_dict.items()]
ax.legend(handles=legend_elements, title="Market Status", loc='lower right', fontsize=15, title_fontsize=16)

plt.tight_layout()
plt.savefig('USA_National_Valuation_Audit.png', dpi=300, bbox_inches='tight')
print("National map saved! This might take a minute to render...")
plt.show()