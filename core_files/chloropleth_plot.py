import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. Load Your ACTUAL XGBoost Audit Data
# ==========================================
# Load the CSV generated in the previous step
df_results = pd.read_csv('USA_ZIP_VALUATION_AUDIT_2026.csv', dtype={'ZIP': str})

# Normalize column names to match the plotting logic
df_results = df_results.rename(columns={'ZIP': 'ZIP_Code', 'valuation_class': 'Valuation_Status'})

# Ensure 'Fairly Priced' is renamed to 'Fairly Valued' if you prefer that label for the paper
df_results['Valuation_Status'] = df_results['Valuation_Status'].replace('Fairly Priced', 'Fairly Valued')

# ==========================================
# 2. Load and Filter the Geographic Shapefile
# ==========================================
shapefile_path = "/Users/kevinkim/Downloads/DM_project/tl_2025_us_zcta520.shp"

# Optimization: Only load the columns we need to save memory
gdf_zcta = gpd.read_file(shapefile_path)
gdf_zcta = gdf_zcta.rename(columns={'ZCTA5CE20': 'ZIP_Code'})

# Filter for Atlanta Region (ZIPs starting with 303) as requested
# This makes the map render instantly instead of taking 2 minutes
gdf_atlanta = gdf_zcta[gdf_zcta['ZIP_Code'].str.startswith('303')].copy()

# ==========================================
# 3. Merge Geopandas with Your Results
# ==========================================
merged_map = gdf_atlanta.merge(df_results, on='ZIP_Code', how='left')

# ==========================================
# 4. Custom Categorical Plotting
# ==========================================
color_dict = {
    'Undervalued': '#2ecc71',   # Emerald Green
    'Fairly Valued': '#bdc3c7', # Silver/Grey
    'Overvalued': '#e74c3c'     # Crimson Red
}

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot the "Base" (ZIPs in 303 that might not be in our audit)
merged_map.plot(ax=ax, color='#f5f5f5', edgecolor='white', linewidth=0.5)

# Plot each category using the actual model results
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

# ==========================================
# 5. Formatting for IEEE Paper
# ==========================================
ax.set_title('Geospatial Valuation Audit: Atlanta Region (2026)', 
             fontdict={'fontsize': 18, 'fontweight': 'bold', 'family': 'serif'})
ax.set_axis_off() 

# Create custom legend handles
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

# Save high-resolution image for LaTeX
plt.tight_layout()
plt.savefig('atlanta_valuation_map.png', dpi=300, bbox_inches='tight')
print("Map successfully saved as 'atlanta_valuation_map.png'")
plt.show()