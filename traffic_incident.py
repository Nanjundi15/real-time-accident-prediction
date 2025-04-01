import folium
from folium.plugins import MarkerCluster, HeatMap
import webbrowser

# List of locations and Node IDs (Example locations in India)
locations = [
    {"lat": 28.6139, "lon": 77.2090, "node_id": 12478830106},  # Delhi
    {"lat": 19.0760, "lon": 72.8777, "node_id": 12478830107},  # Mumbai
    {"lat": 12.9716, "lon": 77.5946, "node_id": 12478830111},  # Bangalore
    {"lat": 22.5726, "lon": 88.3639, "node_id": 12478830116},  # Kolkata
    {"lat": 13.0827, "lon": 80.2707, "node_id": 12478830120},  # Chennai
    {"lat": 23.0225, "lon": 72.5714, "node_id": 12478830123},  # Ahmedabad
    {"lat": 17.385044, "lon": 78.486671, "node_id": 12478830127},  # Hyderabad
    {"lat": 26.8467, "lon": 80.9462, "node_id": 12479146920},  # Lucknow
    {"lat": 27.1751, "lon": 78.0421, "node_id": 12479146926},  # Agra
    {"lat": 30.7333, "lon": 76.7794, "node_id": 12479146931},  # Chandigarh
]

# Create a map centered around India
map_center = [20.5937, 78.9629]  # Center of India coordinates
map = folium.Map(location=map_center, zoom_start=5)

# Marker Clustering to handle overlapping markers
marker_cluster = MarkerCluster().add_to(map)

# List to store coordinates for HeatMap
heat_data = []

# Add markers and prepare heatmap data
for location in locations:
    # Add markers to the map
    folium.Marker(
        location=[location['lat'], location['lon']],
        popup=f"Node ID: {location['node_id']}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(marker_cluster)
    
    # Prepare data for the HeatMap (just latitude and longitude)
    heat_data.append([location['lat'], location['lon']])

# Add HeatMap layer to the map
HeatMap(heat_data).add_to(map)

# Save the map to an HTML file
map.save("india_traffic_incidents_map_with_heatmap.html")

# Open the saved HTML file in the default web browser
webbrowser.open("india_traffic_incidents_map_with_heatmap.html")

print("Map with clustered markers and heatmap has been generated and opened in your browser.")
