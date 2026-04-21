import pandas as pd
import numpy as np
import sys

try:
    df = pd.read_csv('master_schools_2017_2025.csv', dtype={'ZIP': str})
except FileNotFoundError:
    print("Error: 'master_schools_2017_2025.csv' not found.")
    sys.exit()

print("Calculating school counts per ZIP and Academic Year...")


zip_counts = df.groupby(['ZIP', 'SCHOOL_YEAR']).size().reset_index(name='school_count')


pivot_df = zip_counts.pivot(index='ZIP', columns='SCHOOL_YEAR', values='school_count').fillna(0)


yoy_change = pivot_df.diff(axis=1).fillna(0)


yoy_growth = pivot_df.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)


long_counts = pivot_df.reset_index().melt(id_vars='ZIP', value_name='school_count')
long_change = yoy_change.reset_index().melt(id_vars='ZIP', value_name='abs_change')
long_growth = yoy_growth.reset_index().melt(id_vars='ZIP', value_name='pct_growth')


final_metrics = long_counts.merge(long_change, on=['ZIP', 'SCHOOL_YEAR'])
final_metrics = final_metrics.merge(long_growth, on=['ZIP', 'SCHOOL_YEAR'])


final_metrics.to_csv("zip_school_density_yoy.csv", index=False)

print("\n[!] SUCCESS: Created 'zip_school_density_yoy.csv'")
print(f"Total Rows: {len(final_metrics)}")
print(final_metrics.head(10))