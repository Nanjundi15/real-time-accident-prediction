import requests
import folium
import webbrowser
import os  # To get the absolute file path

# ====== Define Bounding Box for Your Area ======
# Format: south, west, north, east (latitude, longitude)
# Example: New York City bounding box
bbox = "40.712, -74.006, 40.740, -73.970"

# Overpass API query to fetch road data
overpass_url = "https://overpass-api.de/api/interpreter"
query = f"""
[out:json];
(
  way[highway]( {bbox} );
  >;
);
out;
"""

# ====== Fetch Data from Overpass API ======
response = requests.get(overpass_url, params={'data': query})
if response.status_code != 200:
    print("Failed to fetch data:", response.status_code)
    exit()

data = response.json()

# ====== Extract Coordinates ======
road_segments = []
for element in data['elements']:
    if 'lat' in element and 'lon' in element:
        road_segments.append((element['lat'], element['lon']))

# ====== Create Folium Map ======
# Center the map around the first road segment
map_center = [road_segments[0][0], road_segments[0][1]]
m = folium.Map(location=map_center, zoom_start=12)

# Add road segments as markers
for lat, lon in road_segments[:100]:
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"Lat: {lat}, Lon: {lon}"
    ).add_to(m)

# ====== Save the Map ======
map_file = "traffic_map.html"
abs_file_path = os.path.abspath(map_file)  # Get full path
m.save(abs_file_path)

# ====== Automatically Open in Browser ======
try:
    webbrowser.open(f"file://{abs_file_path}")  # Open in browser
    print(f"Map saved and opened in your browser: {abs_file_path}")
except Exception as e:
    print(f"Failed to open in browser: {e}")
    
    # Fallback mechanism for different OS
    if os.name == 'nt':  # Windows
        os.startfile(abs_file_path)
    else:  # macOS/Linux
        os.system(f"open {abs_file_path}")
