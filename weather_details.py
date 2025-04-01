import requests

# ====== API Configuration ======
api_key = "8cd0273b24c077ce74c48408382a76b9"  # Your OpenWeatherMap API Key
latitude = 40.7128    # Example: New York City
longitude = -74.0060   # Example: New York City

# ====== OpenWeatherMap API URL ======
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"

# ====== Fetch Weather Data ======
response = requests.get(weather_url)

# ====== Display Weather Info ======
if response.status_code == 200:
    weather_data = response.json()

    # Extracting weather details
    temp = weather_data['main']['temp']
    feels_like = weather_data['main']['feels_like']
    humidity = weather_data['main']['humidity']
    pressure = weather_data['main']['pressure']
    weather_desc = weather_data['weather'][0]['description'].capitalize()
    wind_speed = weather_data['wind']['speed']

    # Displaying the weather info
    print("\n🌦️ Weather Details 🌦️")
    print(f"🌡️ Temperature: {temp}°C (Feels like: {feels_like}°C)")
    print(f"💧 Humidity: {humidity}%")
    print(f"🌬️ Wind Speed: {wind_speed} m/s")
    print(f"📉 Pressure: {pressure} hPa")
    print(f"☁️ Conditions: {weather_desc}")

else:
    print(f"❌ Failed to fetch weather data: {response.status_code}")
    print(response.text)
