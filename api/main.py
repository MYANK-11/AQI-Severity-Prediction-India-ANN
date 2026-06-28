# main.py
# FastAPI backend for AQI Severity Prediction

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
import numpy as np
from model import AQI_ANN

# ---- Initialize FastAPI app ----
app = FastAPI(title="AQI Severity Prediction API")

# ---- Load saved artifacts ----
# Load label encoder first - need it to know num_classes
with open('../models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Recreate model architecture, then load trained weights
input_size = 17  # PM25, PM10, NO2, SO2, CO, Ozone, NH3, RH, WS, WD, Month, Year, 5 City columns
num_classes = len(le.classes_)

model = AQI_ANN(input_size, num_classes)
model.load_state_dict(torch.load('../models/AQI_ANN_model.pth', map_location='cpu'))
model.eval()
# model.eval() is critical here - disables Dropout for inference

# ---- Health advisory messages ----
health_advisory = {
    "Good": "Air quality is excellent. Enjoy outdoor activities freely.",
    "Satisfactory": "Air quality is acceptable. Minor discomfort possible for sensitive individuals.",
    "Moderate": "Sensitive groups (children, elderly, asthma patients) should limit prolonged outdoor exertion.",
    "Poor": "Everyone may experience breathing discomfort. Limit outdoor activities.",
    "Very Poor": "Health warning - avoid outdoor activities. Use masks if going outside.",
    "Severe": "Health emergency. Avoid all outdoor activities. Stay indoors with air purifiers if possible."
}

# ---- Request schema using Pydantic ----
class AQIInput(BaseModel):
    PM25: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    Ozone: float
    NH3: float
    RH: float
    WS: float
    WD: float
    Month: int
    Year: int
    City: str  # Delhi, Mumbai, Bengaluru, Chennai, Kolkata

# ---- Prediction endpoint ----
@app.post("/predict")
def predict_aqi(data: AQIInput):
    
    # Step 1: One-hot encode the City manually
    # Must match training order: Bengaluru, Chennai, Delhi, Kolkata, Mumbai
    cities = ['Bengaluru', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
    city_encoded = [1 if data.City == city else 0 for city in cities]
    
    # Step 2: Build feature array in EXACT same order as training
    features = [
        data.PM25, data.PM10, data.NO2, data.SO2, data.CO,
        data.Ozone, data.NH3, data.RH, data.WS, data.WD,
        data.Month, data.Year
    ] + city_encoded
    
    features_array = np.array(features).reshape(1, -1)
    
    # Step 3: Scale using the SAME scaler from training
    features_scaled = scaler.transform(features_array)
    
    # Step 4: Convert to tensor and predict
    features_tensor = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Step 5: Decode prediction back to category name
    category = le.inverse_transform([predicted_class])[0]
    
    return {
        "AQI_Category": category,
        "Confidence": round(confidence * 100, 2),
        "Health_Advisory": health_advisory[category]
    }

# ---- Root endpoint - health check ----
@app.get("/")
def root():
    return {"message": "AQI Severity Prediction API is running ✅"}