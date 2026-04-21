"""
debug_noaa.py — Run this to inspect raw NOAA API responses for a failing city.
Usage: python3 debug_noaa.py
"""
import requests
import json
import time

NOAA_TOKEN = "YOUR_NOAA_TOKEN_HERE"   
HEADERS    = {"token": NOAA_TOKEN}

STATION_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
DATA_URL    = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

CALL_DELAY = 0.5   

CITY   = "Philadelphia"
LAT    = 39.9526
LON    = -75.1652

START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

def show(label, data):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    print(json.dumps(data, indent=2)[:3000])  

def safe_get(session, url, retries=3, **kwargs):
    """GET with a mandatory delay, timeout retries, and 429 awareness."""
    for attempt in range(1, retries + 1):
        time.sleep(CALL_DELAY)
        try:
            res = session.get(url, **kwargs)
        except requests.exceptions.Timeout:
            print(f"  [!] Timeout (attempt {attempt}/{retries}) — retrying…")
            time.sleep(2 ** attempt)   
            continue
        if res.status_code == 429:
            wait = int(res.headers.get("Retry-After", 60))
            print(f"  [!] 429 rate limited — waiting {wait}s before retrying…")
            time.sleep(wait)
            continue
        return res
    raise RuntimeError(f"All {retries} attempts failed for {url}")

session = requests.Session()
session.headers.update(HEADERS)

for datatypeid in ["TAVG", "TMAX", "TMIN"]:
    for radius in [0.3, 0.75, 1.5]:
        extent = f"{LAT-radius},{LON-radius},{LAT+radius},{LON+radius}"
        print(f"\n>>> Station search: {datatypeid}, radius={radius}°")

        res  = safe_get(session, STATION_URL, params={
            "extent":     extent,
            "datasetid":  "GSOM",
            "datatypeid": datatypeid,
            "limit":      5,
        }, timeout=10)

        body = res.json()
        show(f"Station results ({datatypeid}, r={radius})", body)

        if "results" not in body:
            print("  ↳ No stations found at this radius.")
            continue

        
        for station in body["results"]:
            sid = station["id"]
            print(f"\n  >>> Data fetch: {sid} | mindate={station.get('mindate')} maxdate={station.get('maxdate')}")
            data_res = safe_get(session, DATA_URL, params={
                "datasetid":  "GSOM",
                "datatypeid": datatypeid,
                "stationid":  sid,
                "startdate":  START_DATE,
                "enddate":    END_DATE,
                "limit":      10,
                "units":      "standard",
            }, timeout=15)
            show(f"Data ({datatypeid} @ {sid})", data_res.json())

        break   

print("\n\nDone. Paste the output above to diagnose the issue.")