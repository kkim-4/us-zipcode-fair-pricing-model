import pandas as pd
import requests
import time
import os
import logging
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("noaa_fetch.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


NOAA_TOKEN = "NOAA_TOKEN"
HEADERS = {"token": NOAA_TOKEN}

STATION_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
DATA_URL    = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
OUTPUT_FILE = "noaa_official_monthly_weather.csv"

MAX_CALLS_PER_DAY = 9800
MAX_RETRIES       = 3       
BASE_BACKOFF      = 2       
MAX_BACKOFF       = 120   
CALL_DELAY        = 0.25    

START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

calls_today  = 0
window_start = time.monotonic()
window_calls = 0
WINDOW_SIZE  = 1.0  
WINDOW_LIMIT = 4     

def rate_limited_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """
    Makes a GET request while enforcing:
      - A rolling 1-second window cap (WINDOW_LIMIT calls/sec)
      - Exponential backoff on 429 responses
      - Up to MAX_RETRIES attempts before raising
    """
    global calls_today, window_start, window_calls

    for attempt in range(1, MAX_RETRIES + 1):

        now = time.monotonic()
        elapsed = now - window_start
        if elapsed >= WINDOW_SIZE:
            window_start = now
            window_calls = 0

        if window_calls >= WINDOW_LIMIT:
            sleep_for = WINDOW_SIZE - elapsed + 0.05
            log.debug(f"Window full — sleeping {sleep_for:.2f}s")
            time.sleep(sleep_for)
            window_start = time.monotonic()
            window_calls = 0


        try:
            response = session.get(url, **kwargs)
            calls_today  += 1
            window_calls += 1
        except requests.exceptions.RequestException as e:
            wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
            log.warning(f"Network error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s…")
            time.sleep(wait)
            continue


        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", BASE_BACKOFF ** attempt))
            wait = min(retry_after, MAX_BACKOFF)
            log.warning(f"429 Rate Limited (attempt {attempt}/{MAX_RETRIES}). Waiting {wait}s…")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            wait = min(BASE_BACKOFF ** attempt, MAX_BACKOFF)
            log.warning(f"Server error {response.status_code} (attempt {attempt}/{MAX_RETRIES}). Retrying in {wait}s…")
            time.sleep(wait)
            continue

        time.sleep(CALL_DELAY)
        return response

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {url}")


def sleep_until_tomorrow():
    """Sleeps until midnight + a small buffer to reset the daily quota."""
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    wait = (tomorrow - now).total_seconds()
    log.info(f"Daily limit reached. Sleeping {wait/3600:.1f}h until {tomorrow.strftime('%H:%M')}…")
    time.sleep(wait)


df_cities = pd.read_json("cities.json")

if os.path.exists(OUTPUT_FILE):
    completed_df    = pd.read_csv(OUTPUT_FILE)
    completed_cities = set(completed_df["city"].unique())
    log.info(f"Resuming — {len(completed_cities)} cities already processed.")
else:
    completed_cities = set()
    pd.DataFrame(columns=[
        "city", "state", "latitude", "longitude",
        "station_id", "YearMonth", "Monthly_Avg_Temp"
    ]).to_csv(OUTPUT_FILE, index=False)


log.info(f"Starting NOAA fetch for {len(df_cities)} cities…")
session = requests.Session()
session.headers.update(HEADERS)

skipped  = []
failures = []

for _, row in df_cities.iterrows():
    city_name = row["city"]

    if city_name in completed_cities:
        continue

    if calls_today >= MAX_CALLS_PER_DAY:
        sleep_until_tomorrow()
        calls_today = 0

    log.info(f"[{calls_today} calls] Processing: {city_name}, {row['state']}")

    lat, lon = float(row["latitude"]), float(row["longitude"])


    def coverage_score(s):
        try:
            start        = datetime.fromisoformat(s.get("mindate", "9999-01-01"))
            end          = datetime.fromisoformat(s.get("maxdate", "1900-01-01"))
            target_start = datetime.fromisoformat(START_DATE)
            target_end   = datetime.fromisoformat(END_DATE)
            return (min(end, target_end) - max(start, target_start)).days
        except Exception:
            return 0

    def get_candidates(datatypeid: str) -> list[str]:
        """Return up to 5 station IDs for the given datatype, widening radius if needed."""
        for search_radius in [0.3, 0.75]:
            extent = (
                f"{lat - search_radius},{lon - search_radius},"
                f"{lat + search_radius},{lon + search_radius}"
            )
            try:
                res  = rate_limited_get(
                    session, STATION_URL,
                    params={"extent": extent, "datasetid": "GSOM",
                            "datatypeid": datatypeid, "limit": 5},
                    timeout=10,
                )
                body = res.json()
                if res.status_code == 200 and "results" in body:
                    ranked = sorted(body["results"], key=coverage_score, reverse=True)
                    ids    = [s["id"] for s in ranked]
                    log.debug(f"Candidates for {datatypeid} ({search_radius}°): {ids}")
                    return ids
            except RuntimeError as e:
                log.error(f"Station lookup failed for {city_name}/{datatypeid}: {e}")
                return []
        return []


    def fetch_datatype(datatypeid: str, candidate_ids: list[str]) -> pd.DataFrame | None:
        """
        Try every candidate station and return the DataFrame with the
        most rows, or None if all come back empty.
        """
        best_df    = None
        best_rows  = 0

        for station_id in candidate_ids:
            log.debug(f"Trying {datatypeid} @ {station_id} for {city_name}…")
            try:
                res  = rate_limited_get(
                    session, DATA_URL,
                    params={
                        "datasetid":  "GSOM",
                        "datatypeid": datatypeid,
                        "stationid":  station_id,
                        "startdate":  START_DATE,
                        "enddate":    END_DATE,
                        "limit":      1000,
                        "units":      "standard",
                    },
                    timeout=15,
                )
                body = res.json()
                if res.status_code == 200 and "results" in body:
                    n = len(body["results"])
                    if n > best_rows:
                        df            = pd.DataFrame(body["results"])
                        df["station"] = station_id
                        best_df       = df
                        best_rows     = n
                        log.debug(f"  New best: {n} rows from {station_id}")
            except RuntimeError as e:
                log.error(f"Data fetch failed for {city_name}/{station_id}: {e}")

        return best_df


    tavg_candidates = get_candidates("TAVG")
    best_df         = fetch_datatype("TAVG", tavg_candidates) if tavg_candidates else None


    if best_df is None or len(best_df) < 24:   
        log.info(f"TAVG sparse/missing for {city_name} — trying TMAX+TMIN fallback…")

        tmax_candidates = get_candidates("TMAX")
        tmin_candidates = get_candidates("TMIN")

        df_tmax = fetch_datatype("TMAX", tmax_candidates) if tmax_candidates else None
        df_tmin = fetch_datatype("TMIN", tmin_candidates) if tmin_candidates else None

        if df_tmax is not None and df_tmin is not None:

            tmax = df_tmax[["date", "value"]].rename(columns={"value": "TMAX"})
            tmin = df_tmin[["date", "value"]].rename(columns={"value": "TMIN"})
            merged = tmax.merge(tmin, on="date", how="inner")
            if not merged.empty:
                merged["value"]   = (merged["TMAX"] + merged["TMIN"]) / 2
                merged["station"] = df_tmax["station"].iloc[0]
                best_df           = merged
                log.info(f"TMAX+TMIN fallback yielded {len(merged)} rows for {city_name}.")


    if best_df is not None and len(best_df) >= 12: 
        winning_station = best_df["station"].iloc[0]
        df_clean = pd.DataFrame({
            "city":             city_name,
            "state":            row["state"],
            "latitude":         lat,
            "longitude":        lon,
            "station_id":       winning_station,
            "YearMonth":        pd.to_datetime(best_df["date"]).dt.to_period("M").dt.to_timestamp(),
            "Monthly_Avg_Temp": best_df["value"],
        })
        df_clean[["city", "state", "latitude", "longitude",
                  "station_id", "YearMonth", "Monthly_Avg_Temp"]
        ].to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        log.info(f"Saved {len(df_clean)} rows for {city_name} via {winning_station}.")
    else:
        log.warning(f"No usable data found for {city_name}. Marking skipped.")
        skipped.append(city_name)


log.info("\n========== FETCH COMPLETE ==========")
log.info(f"Total API calls made : {calls_today}")
log.info(f"Cities skipped       : {len(skipped)}  → {skipped}")
log.info(f"Cities failed        : {len(failures)} → {failures}")
log.info(f"Output saved to      : {OUTPUT_FILE}")