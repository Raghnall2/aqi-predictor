from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from ml import inferance_model1
from ml import train_model
import os
import pickle

app = FastAPI()
scaler1= os.path.join(r"model/scaler.pkl")
with open(scaler1, "rb") as file:
    scaler = pickle.load(file)

# ── Input Schema ──────────────────────────────────────────
class AQIInput(BaseModel):
    pollutant_min : float
    pollutant_avg : float
    state         : int   # label encoded
    city          : int   # label encoded
    pollutant_id  : int   # label encoded
    latitude      : float
    longitude     : float

# ── Health Check ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AQI Predictor API is running"}

# ── Predict Endpoint ──────────────────────────────────────
@app.post("/predict")
def predict(data: AQIInput):
    input_data = np.array([[
        data.pollutant_min,
        data.pollutant_avg,
        data.state,
        data.city,
        data.pollutant_id,
        data.latitude,
        data.longitude
    ]])
     
    scaled_input=scaler.transform(input_data)


    
    prediction = inferance_model1(scaled_input)
    
    return {
        "pollutant_max_prediction": round(float(prediction[0]), 2)
    }

# ── Metrics Endpoint ──────────────────────────────────────
@app.get("/metrics")
def metrics():
    with open("model/metrices.json", "r") as f:
        return json.load(f)


