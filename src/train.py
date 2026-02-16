import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_processing import load_data, preprocess_data, get_train_test_data
from model import VirtualInfluencerNet
import os
#hii
def train_model():
    # 1. Load and Process Data
    print("Loading data...")
    df = load_data()
    if df is None:
        return
    
    processed_df, target_cols = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = get_train_test_data(processed_df, target_cols)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) # Shape: (batch, 4)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32) # Shape: (batch, 4)
    
    # 2. Initialize Model
    input_size = X_train.shape[1]
    output_size = len(target_cols)
    model = VirtualInfluencerNet(input_size=input_size, output_size=output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 200
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss (MSE): {test_loss.item():.4f}')
        print(f'Test RMSE: {np.sqrt(test_loss.item()):.4f}')
        
    # 5. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/vi_model.pth')
    print("Model saved to models/vi_model.pth")
    
    # Save Scaler for inference
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Save Feature Names
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else []
    # Wait, X_train from get_train_test_data is numpy array because of StandardScaler? No, get_train_test_data returns numpy arrays.
    # We need to get feature names BEFORE scaling or modify get_train_test_data to return df.
    # In get_train_test_data: "X_scaled = scaler.fit_transform(X)". X is df_numeric.drop(target).
    
    X_columns = processed_df.drop(target_cols, axis=1).columns.tolist()
    
    import json
    # Save features AND targets config
    config = {
        'features': X_columns,
        'targets': target_cols
    }
    with open('models/config.json', 'w') as f:
        json.dump(config, f)
    print("Model config (features + targets) saved to models/config.json")

if __name__ == "__main__":
    train_model()
