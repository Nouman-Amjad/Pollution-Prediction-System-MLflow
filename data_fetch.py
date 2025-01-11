import requests
import json
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and Flask endpoint from .env
API_KEY = os.getenv('API_KEY')  # OpenWeatherMap API key
PREDICT_API_URL = "http://localhost:5000/predict"  # Flask prediction endpoint

# Coordinates for the desired location (Example: Multiple cities)
coordinates = [
    {"lat": 33.6938118, "lon": 73.0651511},  # Islamabad
    {"lat": 40.7128, "lon": -74.0060},       # New York
    {"lat": 51.5074, "lon": -0.1278}         # London
]

# Output file
DATA_FILE = 'Data/live_environmental_predictions.csv'

# Function to fetch air pollution data
def fetch_air_pollution_data(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        pollution_data = data['list'][0]  # Access the first record
        return {
            "AQI": pollution_data["main"]["aqi"],
            "CO": pollution_data["components"]["co"],
            "NO2": pollution_data["components"]["no2"],
            "PM2.5": pollution_data["components"]["pm2_5"],
            "PM10": pollution_data["components"]["pm10"],
            "Ozone": pollution_data["components"]["o3"],
        }
    else:
        print(f"[Error] Failed to fetch air pollution data: {response.status_code}")
        return None

# Function to fetch weather data
def fetch_weather_data(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "Wind Speed": data["wind"]["speed"],
            "Pressure": data["main"]["pressure"],
            "Weather Description": data["weather"][0]["description"],
        }
    else:
        print(f"[Error] Failed to fetch weather data: {response.status_code}")
        return None

# Function to send data to the prediction API
def get_prediction(features):
    payload = {"features": [features]}
    try:
        response = requests.post(PREDICT_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("prediction", None)
        else:
            print(f"[Error] Prediction failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[Error] Failed to connect to prediction API: {e}")
        return None

# Main Loop for Continuous Data Fetching
def main():
    print("Starting live data collection and prediction testing...")
    while True:
        all_data = []
        for coord in coordinates:
            lat, lon = coord["lat"], coord["lon"]
            print(f"Fetching data for Latitude: {lat}, Longitude: {lon}...")
            
            # Fetch pollution and weather data
            pollution_data = fetch_air_pollution_data(lat, lon)
            weather_data = fetch_weather_data(lat, lon)
            
            if pollution_data and weather_data:
                combined_data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Latitude": lat,
                    "Longitude": lon,
                    **pollution_data,
                    **weather_data,
                }
                
                # Prepare features for the prediction API
                features = [
                    combined_data["Temperature"],
                    combined_data["Humidity"],
                    combined_data["Pressure"],
                    combined_data["Wind Speed"],
                    combined_data["PM2.5"]
                ]
                
                # Get prediction
                prediction = get_prediction(features)
                if prediction is not None:
                    combined_data["Prediction"] = prediction[0]  # Append prediction result
                else:
                    combined_data["Prediction"] = "N/A"

                all_data.append(combined_data)
                print(f"Data Collected: {combined_data}")
        
        # Save data to CSV file
        try:
            if not os.path.exists(DATA_FILE):
                pd.DataFrame(all_data).to_csv(DATA_FILE, index=False)
            else:
                pd.DataFrame(all_data).to_csv(DATA_FILE, mode='a', header=False, index=False)
            print(f"Data successfully saved to {DATA_FILE}")
        except Exception as e:
            print(f"[Error] Failed to save data to CSV: {e}")
        
        print("Sleeping for 5 minutes before next fetch...\n")
        time.sleep(300)  # Wait for 5 minutes

if __name__ == "__main__":
    main()
