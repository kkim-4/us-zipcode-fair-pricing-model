import pandas as pd
import requests
import time
import os
import math
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
NOAA_TOKEN = "AeRKiUhRNGkDvTlIbwEERVFEgXHTtWqo" 
HEADERS = {"token": NOAA_TOKEN}

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
DATA_URL = f"{BASE_URL}/data"
SEARCH_URL = f"{BASE_URL}/stations"

OUTPUT_FILE = "noaa_anchor_seasonal_weather.csv"
START_DATE, END_DATE = "2016-01-01", "2025-12-31" 
CALL_DELAY = 1.2 

def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

# ==========================================
# 2. LOAD & RESUME
# ==========================================
df_anchors = pd.read_json('weather_anchors.json', dtype={'zip': str})
df_anchors = df_anchors[~df_anchors['state_id'].isin(['PR', 'GU', 'VI', 'AS', 'MP'])].copy()

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 100:
    completed_df = pd.read_csv(OUTPUT_FILE, dtype={'ZIP_Code': str})
    completed_zips = set(completed_df['ZIP_Code'].unique())
    print(f"Resuming: {len(completed_zips)} ZIPs finished.")
else:
    completed_zips = set()
    cols = ['ZIP_Code', 'city', 'state_id', 'station_id', 'YearMonth', 'TAVG']
    pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)

# ==========================================
# 3. ROBUST FETCH ENGINE
# ==========================================
print(f"Starting Robust Scrape. Shields active. Press Ctrl+C to stop.\n")

try:
    for i, (_, row) in enumerate(df_anchors.iterrows()):
        zip_code = row['zip']
        if zip_code in completed_zips: continue

        print(f"[{i+1}/{len(df_anchors)}] ZIP {zip_code} ({row['city']})")

        # STEP 1: Search for stations (with error shield)
        s_params = {
            "extent": f"{row['lat']-0.8},{row['lng']-0.8},{row['lat']+0.8},{row['lng']+0.8}",
            "datasetid": "GSOM",
            "datatypeid": "TAVG",
            "limit": 50
        }
        
        s_res = None
        for attempt in range(3): # Try 3 times before skipping
            time.sleep(CALL_DELAY)
            try:
                resp = requests.get(SEARCH_URL, headers=HEADERS, params=s_params, timeout=20)
                if resp.status_code == 200:
                    s_res = resp.json()
                    break
                else:
                    print(f"   [!] Attempt {attempt+1}: Server returned {resp.status_code}. Retrying...")
                    time.sleep(5)
            except Exception as e:
                print(f"   [!] Attempt {attempt+1}: Connection issue. Retrying...")
                time.sleep(5)

        if not s_res or 'results' not in s_res:
            print(f"   [-] No temperature-certified stations found for {zip_code}.")
            continue
        
        candidates = []
        for s in s_res['results']:
            if s['mindate'] <= '2017-01-01' and s['maxdate'] >= '2024-01-01':
                s['dist'] = calculate_distance(row['lat'], row['lng'], s['latitude'], s['longitude'])
                s['priority'] = 0 if s['id'].startswith('GHCND:USW') else 1
                candidates.append(s)
        candidates.sort(key=lambda x: (x['priority'], x['dist']))

        # STEP 2: Pull Data (with error shield)
        data_secured = False
        for station in candidates[:5]:
            print(f"   Checking {station['id']}...", end="\r")
            
            d_params = {
                "datasetid": "GSOM",
                "stationid": station['id'],
                "units": "standard",
                "startdate": START_DATE,
                "enddate": END_DATE,
                "datatypeid": ["TAVG", "TMAX", "TMIN"],
                "limit": 1000
            }

            try:
                time.sleep(CALL_DELAY)
                d_resp = requests.get(DATA_URL, headers=HEADERS, params=d_params, timeout=25)
                if d_resp.status_code != 200: continue
                
                data = d_resp.json()
                if 'results' not in data: continue
                
                df_raw = pd.DataFrame(data['results'])
                df_pivot = df_raw.pivot_table(index='date', columns='datatype', values='value').reset_index()
                
                temp_col = 'TAVG' if 'TAVG' in df_pivot.columns else None
                if not temp_col and 'TMAX' in df_pivot.columns and 'TMIN' in df_pivot.columns:
                    df_pivot['T_CALC'] = (df_pivot['TMAX'] + df_pivot['TMIN']) / 2
                    temp_col = 'T_CALC'
                
                if not temp_col: continue

                df_pivot['YearMonth'] = pd.to_datetime(df_pivot['date'])
                df_seasonal = df_pivot[df_pivot['YearMonth'].dt.month.isin([1, 4, 7, 10])].copy()
                
                if not df_seasonal.empty:
                    df_seasonal['ZIP_Code'], df_seasonal['city'], df_seasonal['state_id'] = zip_code, row['city'], row['state_id']
                    df_seasonal['station_id'] = station['id']
                    df_seasonal['TAVG_VAL'] = df_seasonal[temp_col]
                    
                    out_cols = ['ZIP_Code', 'city', 'state_id', 'station_id', 'YearMonth', 'TAVG_VAL']
                    df_seasonal[out_cols].to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    
                    print(f"   SUCCESS: Secured Temperature from {station['id']}      ")
                    data_secured = True
                    break
            except:
                continue
        
        if not data_secured:
            print(f"   FAILED: Could not retrieve data for {zip_code}.")

except KeyboardInterrupt:
    print("\n\n[!] Stopped. Progress saved.")
    sys.exit(0)