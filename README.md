# 🌫️ AQI Severity Prediction for Indian Cities using ANN

> Predicting Air Quality Index (AQI) severity categories — **Good to Severe** — 
> for 5 major Indian cities using a 3-layer Artificial Neural Network (PyTorch) 
> trained on **9,399 real CPCB sensor records** spanning 2019–2024.

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Overall Validation Accuracy | **85.27%** |
| Macro F1-Score | **0.86** |
| F1-Score on Critical Severe Days | **0.92** |
| Training Epochs (Early Stopping) | **29 / 100** |
| Dataset Size | **9,399 daily records** |
| Cities Covered | **5 major Indian cities** |
| Years Covered | **2019 – 2024** |

---

## 🎯 Problem Statement

Air pollution is a public health crisis in India —
**6 of the world's top 10 most polluted cities are Indian.**
Existing AQI systems report pollution *after* it happens.
This project builds a deep learning model that predicts AQI severity category
from raw pollutant concentrations, enabling proactive health alerts
and policy interventions.

---

## 🗃️ Dataset

- **Source:** Central Pollution Control Board (CPCB) — 
  [CAAQMS Data Repository](https://airquality.cpcb.gov.in)
- **Type:** Real government sensor data — not pre-cleaned Kaggle datasets
- **Stations:**

| City | Station |
|------|---------|
| Delhi | Anand Vihar (DPCC) |
| Mumbai | Bandra (MPCB) |
| Bengaluru | Silk Board (KSPCB) |
| Chennai | Manali (CPCB) |
| Kolkata | Rabindra Sarobar (WBPCB) |

- **Features:** PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, 
  Relative Humidity, Wind Speed, Wind Direction, Month, Year, City
- **Target:** AQI Category (6 classes: Good, Satisfactory, 
  Moderate, Poor, Very Poor, Severe)
- **AQI Calculation:** Implemented official CPCB sub-index formula 
  from raw pollutant concentrations

---

## 🔬 EDA Key Insights

📍 **Delhi dominates pollution** — Average AQI of 261, 
exceeding the Moderate threshold by 30%

📅 **Strong seasonality confirmed** — Delhi AQI spikes to 400+ 
in Nov–Jan (winter) and drops to ~130 in Jul–Aug (monsoon), 
validating Month as a powerful model feature

🦠 **COVID-19 lockdown signature detected** — 
AQI dropped ~50% during Apr–Jun 2020 vs same period in 2019, 
validating data authenticity

⚖️ **Class imbalance handled** — Satisfactory class (38.2%) 
vs Severe class (4.4%) addressed using class-weighted loss function

---

## 🏗️ Project Pipeline
