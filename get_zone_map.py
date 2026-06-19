import requests
import json

url = "https://services.nyserda.ny.gov/arcgis/rest/services/Electric/Utility_and_Load_Zones/MapServer/0/query"
params = {
    "where": "1=1",
    "outFields": "*",
    "f": "geojson"
}

response = requests.get(url, params=params)
data = response.json()

print(f"Number of features: {len(data.get('features', []))}")
if data.get('features'):
    print("Sample feature properties:")
    print(json.dumps(data['features'][0]['properties'], indent=2))

with open("ny_load_zones.geojson", "w") as f:
    json.dump(data, f)

print("\nSaved ny_load_zones.geojson")