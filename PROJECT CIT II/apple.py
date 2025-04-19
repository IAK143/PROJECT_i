import requests


def get_current_location():
    """
    Attempts to determine your current latitude and longitude using your IP address.
    This method uses the ipinfo.io service; note that accuracy can vary.
    """
    try:
        response = requests.get("https://ipinfo.io/json")
        if response.status_code == 200:
            data = response.json()
            loc_str = data.get("loc")
            if loc_str:
                lat, lon = map(float, loc_str.split(","))
                return lat, lon
        print("Unable to determine location via IP; using fallback coordinates.")
    except Exception as e:
        print("Error while fetching location data:", e)
    # Fallback: Use approximate coordinates (Sahid Matangini, West Bengal, India)
    return 22.5998, 88.3933


def get_pollution_data(lat, lon, api_key):
    """
    Retrieves air pollution data for the provided latitude and longitude using OpenWeatherMap's API.
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error fetching pollution data. HTTP Status Code:", response.status_code)
    except Exception as e:
        print("Error occurred while fetching pollution data:", e)
    return None


def parse_and_display(data):
    """
    Parses the JSON response from OpenWeatherMap and prints out the Air Quality Index (AQI)
    along with individual pollutant measures.
    """
    if not data or "list" not in data or len(data["list"]) == 0:
        print("No pollution data available.")
        return

    pollution_info = data["list"][0]
    aqi = pollution_info["main"]["aqi"]
    components = pollution_info["components"]

    # Mapping Air Quality Index to human-readable descriptions
    aqi_descriptions = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }

    print(f"Air Quality Index (AQI): {aqi} ({aqi_descriptions.get(aqi, 'Unknown')})")
    print("Pollutant Concentrations (in μg/m³):")
    for pollutant, value in components.items():
        print(f" - {pollutant.upper()}: {value}")


def main():
    # Get current coordinates (latitude and longitude)
    lat, lon = get_current_location()
    print(f"Using coordinates: Latitude = {lat}, Longitude = {lon}")

    # API key provided by you
    api_key = "d88a5d9c25b36002625f24baa0738917"

    # Retrieve the pollution data
    data = get_pollution_data(lat, lon, api_key)
    if data:
        print("\nPollution Data:")
        parse_and_display(data)
    else:
        print("Failed to retrieve pollution data.")


if __name__ == "__main__":
    main()
