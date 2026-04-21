import pandas as pd
import numpy as np
import sys

# 1. Load the compiled file
# Ensure ZIP is read as a string to preserve leading zeros
try:
    df = pd.read_csv('master_schools_2017_2025.csv', dtype={'ZIP': str})
except FileNotFoundError:
    print("Error: 'master_schools_2017_2025.csv' not found.")
    sys.exit()

print("Calculating school counts per ZIP and Academic Year...")

# 2. Group by ZIP and SCHOOL_YEAR to get the count of schools
# The column name from your data is 'SCHOOL_YEAR'
zip_counts = df.groupby(['ZIP', 'SCHOOL_YEAR']).size().reset_index(name='school_count')

# 3. Pivot the data to 'Wide' format
# This puts years (1718, 1819, 1920, etc.) as column headers
pivot_df = zip_counts.pivot(index='ZIP', columns='SCHOOL_YEAR', values='school_count').fillna(0)

# 4. Calculate Year-Over-Year (YoY) absolute change
# This subtracts the count of the previous year from the current year
yoy_change = pivot_df.diff(axis=1).fillna(0)

# 5. Calculate YoY Percentage Growth
yoy_growth = pivot_df.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)

# 6. Melt the data back into a clean 'Machine Learning Ready' format
# We'll merge the counts, changes, and growth back together
long_counts = pivot_df.reset_index().melt(id_vars='ZIP', value_name='school_count')
long_change = yoy_change.reset_index().melt(id_vars='ZIP', value_name='abs_change')
long_growth = yoy_growth.reset_index().melt(id_vars='ZIP', value_name='pct_growth')

# Merge all metrics into one final DataFrame
final_metrics = long_counts.merge(long_change, on=['ZIP', 'SCHOOL_YEAR'])
final_metrics = final_metrics.merge(long_growth, on=['ZIP', 'SCHOOL_YEAR'])

# 7. Save the metrics
final_metrics.to_csv("zip_school_density_yoy.csv", index=False)

print("\n[!] SUCCESS: Created 'zip_school_density_yoy.csv'")
print(f"Total Rows: {len(final_metrics)}")
print(final_metrics.head(10))