import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt

# Path to the trained model
model_file_path = 'aqi_model.pkl'  # Update with your actual model file path

# Replace with your actual API key
API_KEY = "c94c4a330e088154840ee941a3a90bad"

# Function to fetch AQI data
def fetch_aqi_data():
    today = datetime.utcnow()
    three_days_ago = today - timedelta(days=3)
    three_days_ahead = today + timedelta(days=3)

    # Convert to Unix timestamps
    start_unix_time = int(three_days_ago.timestamp())
    end_unix_time = int(three_days_ahead.timestamp())

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8546842&lon=67.0207055&start={start_unix_time}&end={end_unix_time}&appid={API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching data: {response.status_code}")
        return None

    try:
        raw = response.json()
    except ValueError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

    if "list" not in raw:
        st.error("Error: 'list' key not found in the response.")
        return None

    aqi_df = pd.json_normalize(raw["list"])
    return aqi_df

# Fetch AQI data
aqi_data = fetch_aqi_data()

if aqi_data is not None:
    if 'dt' in aqi_data.columns:
        aqi_data['timestamp'] = pd.to_datetime(aqi_data['dt'], unit='s')
        aqi_data.set_index('timestamp', inplace=True)
        aqi_data.drop(columns=['dt'], inplace=True)

        aqi_data['hour'] = aqi_data.index.hour
        aqi_data['day'] = aqi_data.index.day
        aqi_data['month'] = aqi_data.index.month
        aqi_data['day_of_week'] = aqi_data.index.dayofweek
        aqi_data['season'] = aqi_data['month'].apply(lambda x: (x % 12 + 3) // 3)

        for col in ["main.aqi", "components.co", "components.no", "components.no2", 
                    "components.o3", "components.so2", "components.pm2_5", 
                    "components.pm10", "components.nh3"]:
            aqi_data[f"{col.split('.')[-1]}_change_rate"] = aqi_data[col].diff()

        st.title("Karachi AQI Prediction")
        st.subheader("Historical AQI Data")
        st.write(aqi_data)

        # Generate timestamps for the past 3 days and the next 3 days
        past_timestamps = pd.date_range(
            start=aqi_data.index.max() - timedelta(days=3),
            end=aqi_data.index.max(),
            freq='4H'
        )
        future_timestamps = pd.date_range(
            start=aqi_data.index.max() + timedelta(hours=4),
            end=aqi_data.index.max() + timedelta(days=3),
            freq='4H'
        )

        # Combine past and future timestamps
        all_timestamps = past_timestamps.union(future_timestamps)

        # Prepare features DataFrame
        all_features = pd.DataFrame({'timestamp': all_timestamps})
        all_features['hour'] = all_features['timestamp'].dt.hour
        all_features['day'] = all_features['timestamp'].dt.day
        all_features['month'] = all_features['timestamp'].dt.month
        all_features['day_of_week'] = all_features['timestamp'].dt.dayofweek
        all_features['season'] = all_features['month'].apply(lambda x: (x % 12 + 3) // 3)

        # Use the last available data row as default values for other features
        default_values = aqi_data.iloc[-1]
        for col in [
            "main.aqi", "components.co", "components.no", "components.no2",
            "components.o3", "components.so2", "components.pm2_5",
            "components.pm10", "components.nh3", "aqi_change_rate",
            "co_change_rate", "no_change_rate", "no2_change_rate",
            "o3_change_rate", "so2_change_rate", "pm2_5_change_rate",
            "pm10_change_rate", "nh3_change_rate"
        ]:
            if col in aqi_data.columns:
                all_features[col] = default_values[col]

        # Load the model and predict AQI
        relevant_columns = [
            'hour', 'day', 'month', 'day_of_week', 'season', 
            'main.aqi', 'components.co', 'components.no', 'components.no2', 
            'components.o3', 'components.so2', 'components.pm2_5', 
            'components.pm10', 'components.nh3', 'aqi_change_rate', 
            'co_change_rate', 'no_change_rate', 'no2_change_rate', 
            'o3_change_rate', 'so2_change_rate', 'pm2_5_change_rate', 
            'pm10_change_rate', 'nh3_change_rate'
        ]

        try:
            if os.path.isfile(model_file_path):
                model = joblib.load(model_file_path)
                predicted_aqi = model.predict(all_features[relevant_columns].values)
                all_features['Predicted AQI'] = predicted_aqi

                st.subheader("Predicted AQI for the Past 3 Days and Next 3 Days")
                st.write(all_features[['timestamp', 'Predicted AQI']])

                # Plot predictions
                plt.figure(figsize=(12, 6))
                plt.plot(all_features["timestamp"], all_features["Predicted AQI"], label="Predicted AQI", marker="o", color="b")
                plt.title("Predicted AQI for Past and Future Days")
                plt.xlabel("Date & Time")
                plt.ylabel("Predicted AQI Level")
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                # Add download button for predictions
                csv_data = all_features[['timestamp', 'Predicted AQI']].to_csv(index=False)
                st.download_button(
                    label="Download Predicted AQI Data as CSV",
                    data=csv_data,
                    file_name="predicted_aqi_past_and_future_3_days.csv",
                    mime="text/csv"
                )
            else:
                st.error("Model file not found or invalid.")
        except Exception as e:
            st.error(f"Error loading or predicting with the model: {e}")
    else:
        st.error("Error: 'dt' column not found in the data.")
else:
    st.error("No AQI data available.")
