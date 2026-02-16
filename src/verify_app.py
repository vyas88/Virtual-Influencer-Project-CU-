import joblib
import torch
import json
import pandas as pd
import numpy as np
from model import VirtualInfluencerNet
from app import run_inference
from data_processing import preprocess_data, load_data

def verify():
    print("Verifying Model Loading...")
    try:
        scaler = joblib.load('models/scaler.pkl')
        with open('models/features.json', 'r') as f:
            features = json.load(f)
        
        input_size = len(features)
        model = VirtualInfluencerNet(input_size=input_size)
        model.load_state_dict(torch.load('models/vi_model.pth'))
        model.eval()
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    print("Verifying Data Processing...")
    try:
        df = load_data('data/survey_data.xlsx')
        processed_df, target_col = preprocess_data(df)
        print(f"Data processed. Shape: {processed_df.shape}")
    except Exception as e:
        print(f"FAILED to process data: {e}")
        return

    print("Verifying Inference...")
    try:
        predictions = run_inference(model, scaler, features, processed_df)
        print(f"Inference successful. Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5]}")
    except Exception as e:
        print(f"FAILED to run inference: {e}")
        return

    print("VERIFICATION COMPLETE: SUCCESS")

if __name__ == "__main__":
    verify()
