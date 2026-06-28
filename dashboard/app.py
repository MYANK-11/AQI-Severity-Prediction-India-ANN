# app.py
# Streamlit dashboard for AQI Severity Prediction

import streamlit as st
import requests

# ---- Page Configuration ----
st.set_page_config(
    page_title="AQI Severity Predictor",
    page_icon="🌫️",
    layout="centered"
)

# ---- Title ----
st.title("🌫️ AQI Severity Prediction")
st.markdown("Predict Air Quality severity for Indian cities using a trained ANN model")
st.divider()

# ---- City Selection ----
city = st.selectbox(
    "Select City",
    ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"]
)

# ---- Month Selection ----
month = st.slider("Month", 1, 12, 6)
year = st.number_input("Year", min_value=2019, max_value=2026, value=2024)

st.subheader("Pollutant Concentrations")

# ---- Pollutant Inputs ----
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=100.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=150.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=40.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=10.0)

with col2:
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)
    ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, value=30.0)
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, value=25.0)

st.subheader("Weather Conditions")
col3, col4, col5 = st.columns(3)

with col3:
    rh = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
with col4:
    ws = st.number_input("Wind Speed (m/s)", min_value=0.0, value=1.5)
with col5:
    wd = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0)

st.divider()

# ---- Color mapping for AQI categories ----
category_colors = {
    "Good": "#2ecc71",
    "Satisfactory": "#a8e063",
    "Moderate": "#f39c12",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad"
}

# ---- Predict Button ----
if st.button("🔍 Predict AQI Severity", use_container_width=True):
    
    # Build the payload matching our API schema
    payload = {
        "PM25": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
        "CO": co, "Ozone": ozone, "NH3": nh3,
        "RH": rh, "WS": ws, "WD": wd,
        "Month": month, "Year": year, "City": city
    }
    
    try:
        # Call our FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            category = result["AQI_Category"]
            confidence = result["Confidence"]
            advisory = result["Health_Advisory"]
            
            color = category_colors.get(category, "#333333")
            
            # Display result with colored box
            st.markdown(
                f"""
                <div style="background-color:{color}; padding:20px; 
                            border-radius:10px; text-align:center;">
                    <h2 style="color:white; margin:0;">AQI Category: {category}</h2>
                    <p style="color:white; font-size:18px; margin:5px 0;">
                        Confidence: {confidence}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.warning(f"⚠️ **Health Advisory:** {advisory}")
            
        else:
            st.error("Something went wrong with the prediction. Check the API.")
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the FastAPI server is running on port 8000.")

st.divider()
st.caption("Built with PyTorch, FastAPI, and Streamlit | Trained on real CPCB data (2019-2024)")