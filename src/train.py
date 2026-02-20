import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import joblib
from data_processing import load_data, preprocess_data, get_train_test_data
from model import VirtualInfluencerNet


def train_model():

    #LOAD N PROCESS
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    processed_df, target_cols = preprocess_data(df)

    X_train, X_test, y_train, y_test, scaler = get_train_test_data(
        processed_df, target_cols
    )

    # Numpy arrays to Pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


    # INITIALIZE
    input_size = X_train.shape[1]
    output_size = len(target_cols)
    model = VirtualInfluencerNet(
        input_size=input_size,
        output_size=output_size
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAING
    epochs = 200
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):

        model.train()  # Set model to training mode
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        # Compute loss
        loss = criterion(outputs, y_train_tensor)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    model.eval()  # Evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)

        # Sigmoid to convert logits
        probabilities = torch.sigmoid(test_outputs)

        # Probabilities to binary
        predictions = (probabilities > 0.5).float()
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f"\nTest Loss: {test_loss.item():.4f}")
        print(f"\nTest Accuracy: {accuracy.item():.4f}")
    # MODEL SAVING
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/vi_model.pth')
    print("Model saved to models/vi_model.pth")

    # Save Scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    # Save feature + target configuration
    feature_columns = processed_df.drop(target_cols, axis=1).columns.tolist()

    config = {
        "features": feature_columns,
        "targets": target_cols
    }

    with open('models/config.json', 'w') as f:
        json.dump(config, f)

    print("Model configuration saved to models/config.json")


if __name__ == "__main__":
    train_model()