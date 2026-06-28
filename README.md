# 🌫️ AQI Severity Prediction for Indian Cities using ANN

> Predicting Air Quality Index (AQI) severity categories - **Good to Severe** 
> for 5 major Indian cities using a 3-layer Artificial Neural Network (PyTorch) 
> trained on **9,399 real CPCB sensor records** spanning 2019–2024.

---
## 🚀 Live Demo

**Try the deployed application:** [aqi-severity-predictor.streamlit.app](https://aqi-severity-predictor.streamlit.app)

⚠️ *Note: The backend API runs on a free-tier server and may take ~50 seconds to wake up on the first request after inactivity.*

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
## 🏗️ Model Training Pipeline

```
Raw CPCB Data (30 CSV files, 5 cities × 6 years)
↓
Data Loading & City Tagging
↓
Column Cleaning & Type Fixing
↓
Official CPCB AQI Calculation
↓
Missing Value Treatment (City-wise Median Imputation)
↓
Feature Engineering (Month, Year, One-Hot City Encoding)
↓
Train-Test Split (80/20, Stratified)
↓
StandardScaler Normalization
↓
Class Weight Computation
↓
3-Layer ANN Training (PyTorch) with Early Stopping
↓
Evaluation (Accuracy, F1, Confusion Matrix)
↓
Saved Model Artifacts (.pth, scaler.pkl, label_encoder.pkl)
```

---

## 🚀 Deployment Pipeline

```
Saved Model Artifacts
↓
FastAPI Backend — Loads model, exposes /predict endpoint
↓
Deployed on Render (Public REST API)
↓
Streamlit Dashboard — Sends user input to API via HTTP
↓
Deployed on Streamlit Community Cloud (Public Web App)
↓
Real-time AQI Severity Prediction with Health Advisory
```
---

## 🧠 Model Architecture

```
Input Layer         →  17 features
Hidden Layer 1      →  128 neurons | BatchNorm | ReLU | Dropout(0.3)
Hidden Layer 2      →   64 neurons | BatchNorm | ReLU | Dropout(0.3)
Hidden Layer 3      →   32 neurons | BatchNorm | ReLU | Dropout(0.2)
Output Layer        →    6 neurons | Softmax
```

**Training Config:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class weights
- Batch Size: 32
- Early Stopping: Patience = 10 epochs
- Best model checkpoint saved automatically
---

## 📊 Results

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Good | 0.72 | 0.93 | 0.81 |
| Satisfactory | 0.90 | 0.80 | 0.84 |
| Moderate | 0.91 | 0.84 | 0.88 |
| Poor | 0.71 | 0.92 | 0.80 |
| Very Poor | 0.91 | 0.91 | 0.91 |
| **Severe** | **0.92** | **0.93** | **0.92** |
| **Overall** | | | **0.85** |

### Key Observation
All misclassifications occur between **neighboring AQI categories** 
(e.g., Good↔Satisfactory, Poor↔Moderate) — the model never confuses 
Good with Severe. This is the expected behaviour of a well-trained 
ordinal classifier.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Deep Learning | PyTorch 2.11 |
| ML & Preprocessing | scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Backend API | FastAPI, Uvicorn |
| Frontend Dashboard | Streamlit |
| Deployment | Render (API), Streamlit Community Cloud (Dashboard) |
| Development Environment | Google Colab, VS Code |
| Data Source | CPCB CAAQMS Portal |
---

## 📁 Folder Structure
```
AQI-Severity-Prediction-India-ANN/
│
├── api/
│   ├── main.py                      # FastAPI backend - prediction endpoint
│   ├── model.py                     # ANN architecture definition
│   └── requirements.txt             # API dependencies
│
├── dashboard/
│   ├── app.py                       # Streamlit dashboard
│   └── requirements.txt             # Dashboard dependencies
│
├── data/
│   └── AQI_Clean_Dataset.csv        # Cleaned master dataset
│
├── notebooks/
│   ├── AQI_Data_Pipeline.ipynb      # Data collection, cleaning, EDA
│   └── AQI_ANN_Model.ipynb          # Model training & evaluation
│
├── models/
│   ├── AQI_ANN_model.pth            # Saved PyTorch model weights
│   ├── scaler.pkl                   # StandardScaler for inference
│   └── label_encoder.pkl            # LabelEncoder for class decoding
│
├── plots/
│   ├── Plot1_AQI_Distribution.png
│   ├── Plot2_Citywise_AQI.png
│   ├── Plot3_Monthly_Trend.png
│   ├── Plot4_Correlation_Heatmap.png
│   ├── Plot5_Yearly_Trend.png
│   ├── Plot6_COVID_Impact.png
│   ├── Plot7_Confusion_Matrix.png
│   └── Plot8_Training_Curves.png
│
├── .gitignore
├── LICENSE
└── README.md
```
---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/MYANK-11/AQI-Severity-Prediction-India-ANN.git
```

**2. Open in Google Colab**
- Upload `AQI_Data_Pipeline.ipynb` to Colab
- Download raw data from 
  [CPCB CAAQMS Portal](https://airquality.cpcb.gov.in)
- Run all cells sequentially

**3. Train the model**
- Open `AQI_ANN_Model.ipynb` in Colab
- Load `AQI_Clean_Dataset.csv` from your Drive
- Run all cells sequentially

**4. Use the saved model for inference**

```python
import torch, pickle
from model import AQI_ANN

# Load label encoder and scaler
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Recreate architecture and load trained weights
model = AQI_ANN(input_size=17, num_classes=len(le.classes_))
model.load_state_dict(torch.load('models/AQI_ANN_model.pth', map_location='cpu'))
model.eval()
```


**5. Run the API and Dashboard locally (optional)**

Start the FastAPI backend:
```bash
cd api
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

In a new terminal, start the Streamlit dashboard:
```bash
cd dashboard
pip install -r requirements.txt
python -m streamlit run app.py
```

> **Note:** To run locally, update the API URL inside `dashboard/app.py` 
> from the deployed Render URL back to `http://127.0.0.1:8000/predict`


---

## 👤 Author

**Mayank P. Savani**


## 📌 Project Status

✅ **Phase 1 — Model Development:** Complete
- Data pipeline, EDA, and ANN training (85.27% accuracy)

✅ **Phase 2 — Production Deployment:** Complete
- REST API built with FastAPI, deployed on Render
- Interactive dashboard built with Streamlit, deployed on Streamlit Cloud
- Fully functional end-to-end live system
---

*Data sourced directly from India's Central Pollution Control Board (CPCB) — 
the same data used by the Government of India for official AQI reporting.*

