import pandas as pd
import json


input_file = 'weather_anchors.json'
output_file = 'weather_anchors.csv'

try:

    with open(input_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df['zip'] = df['zip'].astype(str).str.zfill(5)


    df[['zip', 'city', 'state_id', 'lat', 'lng']].to_csv(output_file, index=False)

    print(f"--- Export Complete ---")
    print(f"File saved as: {output_file}")
    print(f"Total ZIPs exported: {len(df)}")
    print(f"\nTop 5 for verification:")
    print(df[['zip', 'city', 'state_id']].head())

except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'. Make sure it's in the same folder as this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")