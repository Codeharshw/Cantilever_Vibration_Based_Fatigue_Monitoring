"""
GRU-based Vibration Analysis for Cantilever Rod Fatigue Detection

This module implements a Gated Recurrent Unit (GRU) neural network for analyzing
vibration data from MPU9250 sensor to detect structural fatigue in cantilever rods.
The model is trained on healthy vibration patterns and uses prediction error as
a fatigue indicator.

Based on concepts from "Time Series Forecasting using Deep Learning" by Ivan Gridin.
"""

import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import MinMaxScaler


class GRU(nn.Module):
    """
    GRU-based time series prediction model.
    
    Args:
        hidden_size (int): Size of GRU hidden state
        in_size (int): Input feature dimension (default: 1)
        out_size (int): Output dimension (default: 1)
    """
    
    def __init__(self, hidden_size, in_size=1, out_size=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input sequence
            h (torch.Tensor, optional): Initial hidden state
            
        Returns:
            tuple: (prediction, final_hidden_state)
        """
        out, h_out = self.gru(x, h)
        last_hidden_state = out[:, -1, :]
        prediction = self.fc(last_hidden_state)
        return prediction, h_out


def sliding_window(data, window_size):
    """
    Create input-output pairs using sliding window approach.
    
    Args:
        data (np.array): Time series data
        window_size (int): Length of input sequence
        
    Returns:
        tuple: (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def main():
    """Main execution function for vibration analysis."""
    
    # Configuration parameters
    WINDOW_SIZE = 64
    HIDDEN_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    print("=== GRU Vibration Analysis for Fatigue Detection ===\n")

    # Data loading and preprocessing
    print("[1/6] Loading and preprocessing data...")
    try:
        df = pd.read_csv('calibrated_mpu9250_data.csv')
        time_series_data = df['ax'].values.reshape(-1, 1)
        print(f"Loaded {len(time_series_data)} data points")
    except FileNotFoundError:
        print("Error: 'calibrated_mpu9250_data.csv' not found in current directory")
        sys.exit(1)

    # Data normalization and windowing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series_data)
    X, y = sliding_window(scaled_data, WINDOW_SIZE)
    
    # Data splitting
    train_size = int(len(X) * TRAIN_RATIO)
    val_size = int(len(X) * VAL_RATIO)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Convert to PyTorch tensors
    tensors = {}
    for name, data in [('X_train', X_train), ('y_train', y_train), 
                      ('X_val', X_val), ('y_val', y_val),
                      ('X_test', X_test), ('y_test', y_test)]:
        tensors[name] = torch.tensor(data, dtype=torch.float32)
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Model initialization
    print(f"\n[2/6] Initializing GRU model (hidden_size={HIDDEN_SIZE})...")
    model = GRU(hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\n[3/6] Training model for {EPOCHS} epochs...")
    best_model_state = None
    min_val_loss = float('inf')
    train_losses, val_losses = [], []

    model.train()
    for epoch in range(EPOCHS):
        # Training phase
        optimizer.zero_grad()
        train_pred, _ = model(tensors['X_train'])
        train_loss = criterion(train_pred, tensors['y_train'])
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(tensors['X_val'])
            val_loss = criterion(val_pred, tensors['y_val'])
        model.train()
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Save best model
        if val_loss.item() < min_val_loss:
            min_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{EPOCHS}: Train Loss={train_loss.item():.4f}, "
                  f"Val Loss={val_loss.item():.4f}")

    # Model evaluation
    print("\n[4/6] Evaluating best model on test set...")
    best_model = GRU(hidden_size=HIDDEN_SIZE)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    with torch.no_grad():
        test_predictions, _ = best_model(tensors['X_test'])

    # Denormalize predictions
    y_test_orig = scaler.inverse_transform(tensors['y_test'].numpy())
    test_pred_orig = scaler.inverse_transform(test_predictions.numpy())
    
    # Calculate prediction error (fatigue indicator)
    print("\n[5/6] Computing prediction error for fatigue analysis...")
    prediction_error = np.abs(y_test_orig - test_pred_orig)
    
    # Performance metrics
    mse = np.mean((y_test_orig - test_pred_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(prediction_error)
    
    print(f"Test Set Performance:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # Visualization
    print("\n[6/6] Generating visualizations...")
    
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction comparison
    plt.subplot(1, 2, 2)
    indices = range(min(1000, len(y_test_orig)))  # Limit for readability
    plt.plot(indices, y_test_orig[indices], label='Actual', alpha=0.7)
    plt.plot(indices, test_pred_orig[indices], label='Predicted', alpha=0.7)
    plt.title('Prediction vs Actual (Sample)')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Fatigue indicator plot
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_error, color='red', alpha=0.7)
    plt.title('Prediction Error - Fatigue Indicator')
    plt.xlabel('Time Step (Test Set)')
    plt.ylabel('Absolute Prediction Error')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Analysis Complete ===")
    print(f"Monitor prediction error trends for fatigue detection.")
    print(f"Increasing error patterns may indicate structural degradation.")


if __name__ == "__main__":
    main()
