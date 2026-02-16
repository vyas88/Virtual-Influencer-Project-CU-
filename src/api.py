
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import json
import pandas as pd
import numpy as np
from src.model import VirtualInfluencerNet

app = FastAPI(title="Virtual Influencer Prediction API")

# Global variables to hold model artifacts
model = None
scaler = None
config = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, config
    try:
        scaler = joblib.load('models/scaler.pkl')
        with open('models/config.json', 'r') as f:
            config = json.load(f)
            
        input_size = len(config['features'])
        output_size = len(config['targets'])
        
        model = VirtualInfluencerNet(input_size=input_size, output_size=output_size)
        model.load_state_dict(torch.load('models/vi_model.pth'))
        model.eval()
        print("Model and artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        raise e

class PredictionInput(BaseModel):
    # Flexible input dict to accommodate changing features
    # Example: {"age": 25, "gender_Male": 1, ...}
    features: dict

class PredictionOutput(BaseModel):
    predictions: dict

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    global model, scaler, config
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Create DataFrame from input
        # Initialize with zeros for missing features (handling sparse input)
        input_dict = {f: 0.0 for f in config['features']}
        
        # Update with provided features
        # Note: Users need to provide one-hot encoded keys if they are raw inputs?
        # Ideally we'd have a raw-to-processed pipeline here, but for simplicity
        # we assume the API consumer (n8n) sends pre-processed or compatible keys.
        # OR: We just map what matches.
        
        for k, v in input_data.features.items():
            if k in input_dict:
                input_dict[k] = float(v)
            # Handle categorical mapping if needed? 
            # For now, let's assume the input matches the training features.
        
        # Interaction Term Logic (Duplicated from data_processing/app logic)
        # Ideally this logic lives in a common module. 
        # But `data_processing.py` works on full DFs. 
        # Let's simple check:
        if 'age_x_usage' in config['features']:
             age = input_dict.get('age', 0)
             usage = input_dict.get('social_media_usage', 0)
             input_dict['age_x_usage'] = age * usage
        
        df = pd.DataFrame([input_dict])
        
        # Reindex just to be sure order is perfect
        df = df.reindex(columns=config['features'], fill_value=0)
        
        # Scale
        X_scaled = scaler.transform(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            preds = model(X_tensor).numpy()[0]
            
        # Map to targets
        result = dict(zip(config['targets'], preds.tolist()))
        
        return PredictionOutput(predictions=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
