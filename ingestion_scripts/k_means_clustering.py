import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

df = pd.read_csv('uszips.csv', dtype={'zip': str})

states = df['state_id'].unique()

anchors = []

print(f"Starting geographic clustering for {len(states)} regions...")

for state in states:

    state_data = df[df['state_id'] == state].copy()
    
    
    n_clusters = min(6, len(state_data))
    
    if n_clusters > 0:
       
        coords = state_data[['lat', 'lng']].values
        
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(coords)
        theoretical_centers = kmeans.cluster_centers_
        
        
        tree = cKDTree(coords)
        _, indices = tree.query(theoretical_centers)
        
       
        state_anchors = state_data.iloc[indices]
        anchors.append(state_anchors)


df_anchors = pd.concat(anchor_zips) if 'anchor_zips' in locals() else pd.concat(anchors)
df_anchors.to_json('weather_anchors.json', orient='records')

print(f"\n[+] Success!")
print(f"Generated {len(df_anchors)} anchor ZIP codes across {len(states)} regions.")
print(df_anchors[['zip', 'city', 'state_id']].head(10))