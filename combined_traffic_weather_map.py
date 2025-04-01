import requests
import folium
from IPython.display import IFrame

# ====== API Keys & Configurations ======
weather_api_key = "8cd0273b24c077ce74c48408382a76b9"  # Your OpenWeatherMap API Key

# ====== Step 1: Fetch Traffic Data from OpenStreetMap (OSM) ======
# Bounding box covering a region in India (Mumbai area)
bbox = "18.889, 72.744, 19.271, 73.062"  # (south, west, north, east)

# Overpass API Query for Roads
overpass_url = "https://overpass-api.de/api/interpreter"
query = f"""
[out:json];
(
  way[highway]( {bbox} );
  >;
);
out;
"""

# Fetch OSM data
response = requests.get(overpass_url, params={'data': query})
osm_data = response.json()

# Extract road coordinates
road_segments = []
for element in osm_data['elements']:
    if 'lat' in element and 'lon' in element:
        road_segments.append((element['lat'], element['lon']))

# ====== Step 2: Fetch Weather Data for Multiple Cities ======
locations = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867}
]

# ====== Create the Combined Map ======
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Add road segments
for lat, lon in road_segments:
    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(m)

# Add weather markers
for loc in locations:
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={loc['lat']}&lon={loc['lon']}&appid={weather_api_key}&units=metric"
    weather_response = requests.get(weather_url)

    if weather_response.status_code == 200:
        weather_data = weather_response.json()

        # Extract weather details
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        weather_desc = weather_data['weather'][0]['description'].capitalize()

        popup_text = f"""
        🌦️ <b>{loc['name']} Weather</b><br>
        🌡️ Temp: {temp}°C<br>
        💧 Humidity: {humidity}%<br>
        🌬️ Wind Speed: {wind_speed} m/s<br>
        ☁️ Conditions: {weather_desc}
        """

        folium.Marker(
            location=[loc['lat'], loc['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="green", icon="cloud")
        ).add_to(m)

# ====== Save & Display Map ======
map_file = "combined_traffic_weather_map.html"
m.save(map_file)

# Display the map inline in Colab
IFrame(map_file, width=1000, height=600)
