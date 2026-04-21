import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

# 1. Load the data (Ensure ZIP is a string to keep leading zeros)
# Image shows columns: 'zip', 'lat', 'lng', 'state_id'
df = pd.read_csv('uszips.csv', dtype={'zip': str})

# 2. Extract unique states (usually 50 states + DC)
states = df['state_id'].unique()

anchors = []

print(f"Starting geographic clustering for {len(states)} regions...")

for state in states:
    # Filter for the specific state
    state_data = df[df['state_id'] == state].copy()
    
    # We want exactly 6 clusters per state (or fewer if the state has < 6 zips)
    n_clusters = min(6, len(state_data))
    
    if n_clusters > 0:
        # Get coordinates for this state
        coords = state_data[['lat', 'lng']].values
        
        # 3. Run K-Means to find the 'theoretical' centers
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(coords)
        theoretical_centers = kmeans.cluster_centers_
        
        # 4. THE SMART MOVE: Snap centers to the nearest real ZIP code
        # Using a KDTree is the fastest way to perform this 1-nearest-neighbor search
        tree = cKDTree(coords)
        _, indices = tree.query(theoretical_centers)
        
        # Collect the actual ZIP code data for these 6 points
        state_anchors = state_data.iloc[indices]
        anchors.append(state_anchors)

# 5. Compile and export
df_anchors = pd.concat(anchor_zips) if 'anchor_zips' in locals() else pd.concat(anchors)
df_anchors.to_json('weather_anchors.json', orient='records')

print(f"\n[+] Success!")
print(f"Generated {len(df_anchors)} anchor ZIP codes across {len(states)} regions.")
print(df_anchors[['zip', 'city', 'state_id']].head(10))