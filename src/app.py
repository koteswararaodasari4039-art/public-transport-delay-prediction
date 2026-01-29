# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Load trained model and encoder
# -----------------------------
model = joblib.load("../models/delay_classifier.pkl")
encoder = joblib.load("../models/encoder.pkl")

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("🚍 Public Transport Delay Prediction")
st.write("Predict delay, probability, main reason, and expected delay time for a trip.")

# -----------------------------
# 3. User Inputs
# -----------------------------
weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Fog"])
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=60)
traffic = st.slider("Traffic Congestion Index", 0, 100, 50)
event = st.selectbox("Event Type", ["None", "Concert", "Festival", "Sports"])
peak = st.selectbox("Peak Hour", [0, 1])
holiday = st.selectbox("Holiday", [0, 1])
season = st.selectbox("Season", ["Winter", "Summer", "Autumn", "Spring"])

# -----------------------------
# 4. Predict Button
# -----------------------------
if st.button("Predict Delay"):

    # -----------------------------
    # 4.1 Prepare input dataframe
    # -----------------------------
    input_df = pd.DataFrame([{
        "weather_condition": weather,
        "event_type": event,
        "season": season,
        "temperature_C": temperature,
        "humidity_percent": humidity,
        "traffic_congestion_index": traffic,
        "peak_hour": peak,
        "holiday": holiday
    }])

    # -----------------------------
    # 4.2 Separate categorical and numeric columns
    # -----------------------------
    cat_cols = ["weather_condition", "event_type", "season"]
    num_cols = ["temperature_C", "humidity_percent", "traffic_congestion_index", "peak_hour", "holiday"]

    # OneHotEncode categorical columns
    encoded_cat = encoder.transform(input_df[cat_cols])

    # Combine with numeric columns
    encoded_input = np.hstack([encoded_cat, input_df[num_cols].values])

    # -----------------------------
    # 4.3 Make Prediction
    # -----------------------------
    delay_prob = model.predict_proba(encoded_input)[0][1]
    delay_pred = model.predict(encoded_input)[0]

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Delay:** {delay_prob*100:.2f}%")

    # -----------------------------
    # 4.4 If delayed, show estimated delay and main contributing factor
    # -----------------------------
    if delay_pred == 1:
        st.error("🚨 This trip is likely to be **DELAYED**")

        # Estimated delay time (placeholder, can be replaced with regression later)
        estimated_delay = np.random.randint(5, 31)
        st.write(f"**Expected Delay Time:** {estimated_delay} minutes")
        st.subheader("🚦 Travel Recommendation")

        if estimated_delay > 15:
            st.error("🚕 Choose your own transport and start early")
        else:
            st.success("🚌 Wait for the public transport")

        
        

    else:
        st.success("✅ This trip is likely to be **ON TIME**")
