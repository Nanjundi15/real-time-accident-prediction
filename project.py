import requests

# Define the Overpass API URL for OSM data
url = "http://overpass-api.de/api/interpreter"

# Define the Overpass query to get certain data (e.g., incidents or road data)
query = """
[out:json];
node["highway"="traffic_signals"](37.6,-122.5,37.9,-122.3);
out;
"""

# Make the GET request to the Overpass API
response = requests.get(url, params={'data': query})

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    for element in data['elements']:
        print(f"Node ID: {element['id']}")
        print(f"Location: {element['lat']}, {element['lon']}")
else:
    print(f"Error: {response.status_code}")
