# GRU Vibration Analysis for Cantilever Rod Experiment
# This script is an adaptation of the code from Chapter 4 of the book
# "Time Series Forecasting using Deep Learning" by Ivan Gridin.
# It is specifically tailored to analyze the 'calibrated_mpu9250_data.csv' file for fatigue detection.

# --- Part 1: Imports ---
# Standard libraries for data handling, math, and plotting
import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries for building the neural network
import torch
import torch.nn as nn
from torch import optim

# Scikit-learn for data normalization
from sklearn.preprocessing import MinMaxScaler

# --- Part 2: The GRU Model Definition (from Chapter 4) ---
# This is our "smart assistant with a whiteboard"
class GRU(nn.Module):
    """
    A Gated Recurrent Unit (GRU) model followed by a Linear layer for prediction.
    """
    def __init__(self, hidden_size, in_size=1, out_size=1):
        super(GRU, self).__init__()
        # The GRU layer, which acts as the memory component of the model.
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True  # This makes data handling easier
        )
        # The final decision-making layer. It takes the GRU's summary
        # (the hidden state) and makes a final numerical prediction.
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        # The GRU processes the input sequence `x` and an optional initial hidden state `h`.
        # It returns `out` (the hidden state for every timestep) and `h_out` (the final hidden state).
        out, h_out = self.gru(x, h)
        
        # We only need the summary from the very last timestep.
        last_hidden_state = out[:, -1, :]
        
        # The final summary is passed to the linear layer to get the prediction.
        prediction = self.fc(last_hidden_state)
        
        # We return the prediction and the final hidden state.
        return prediction, h_out

# --- Part 3: Data Preparation Helper Function ---
def sliding_window(data, window_size):
    """
    Creates input-output pairs from a time series dataset.
    This is a crucial step for framing our problem for supervised learning.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        # The input (X) is a window of `window_size` consecutive data points.
        feature = data[i:i + window_size]
        # The output (y) is the single data point immediately following the window.
        target = data[i + window_size]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

# --- Part 4: Main Script Execution ---
if __name__ == "__main__":

    # --- Parameters (Easy to change and experiment with!) ---
    # `window_size` is the length of the sliding window (how many past steps to look at).
    window_size = 64
    # `hidden_size` is the size of the GRU's memory (the "whiteboard").
    gru_hidden_size = 32
    # `learning_rate` controls how quickly the model learns.
    learning_rate = 0.001
    # `training_epochs` is the number of times we show the entire dataset to the model.
    training_epochs = 50

    print("--- Starting GRU Vibration Analysis ---")

    # --- A. Load and Prepare the Dataset ---
    print(f"\n[1/6] Loading data from 'calibrated_mpu9250_data.csv'...")
    try:
        df = pd.read_csv('calibrated_mpu9250_data.csv')
        # We'll focus on the 'ax' column for this analysis.
        # Reshape is needed for the scaler.
        time_series_data = df['ax'].values.reshape(-1, 1)
    except FileNotFoundError:
        print("Error: 'calibrated_mpu9250_data.csv' not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
        sys.exit()

    # Normalize the data to be between 0 and 1. This helps the model train better.
    scaler = MinMaxScaler()
    scaled_ts = scaler.fit_transform(time_series_data)
    
    # Create our dataset using the sliding window.
    X, y = sliding_window(scaled_ts, window_size)
    print(f"Data prepared with {len(X)} samples.")

    # Split data into training, validation, and testing sets.
    # We train on the "healthy" part to learn the normal pattern.
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Convert numpy arrays to PyTorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # --- B. Initialize the Model, Optimizer, and Loss Function ---
    print(f"\n[2/6] Initializing GRU model (Hidden Size: {gru_hidden_size})...")
    model = GRU(hidden_size=gru_hidden_size)
    
    # We use Mean Squared Error as the loss function for this regression task.
    loss_function = nn.MSELoss()
    # Adam is a popular and effective optimizer.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- C. Train the Model ---
    print(f"\n[3/6] Starting training for {training_epochs} epochs...")
    best_model_state = None
    min_val_loss = float('inf')
    training_loss_history = []
    validation_loss_history = []

    model.train() # Set the model to training mode
    for epoch in range(training_epochs):
        optimizer.zero_grad() # Reset gradients
        
        # Forward pass: get predictions on training data
        predictions, _ = model(X_train)
        loss = loss_function(predictions, y_train)
        
        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation data
        with torch.no_grad():
            val_predictions, _ = model(X_val)
            val_loss = loss_function(val_predictions, y_val)
        
        training_loss_history.append(loss.item())
        validation_loss_history.append(val_loss.item())

        # Save the best model based on validation loss
        if val_loss.item() < min_val_loss:
            min_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{training_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    print("Training finished.")

    # --- D. Evaluate the Best Model on the Test Set ---
    print("\n[4/6] Evaluating the best model on the test set...")
    # Load the best model weights
    best_model = GRU(hidden_size=gru_hidden_size)
    best_model.load_state_dict(best_model_state)
    best_model.eval() # Set the model to evaluation mode

    with torch.no_grad():
        test_predictions, _ = best_model(X_test)

    # Denormalize the data to see the results in the original scale
    y_test_unscaled = scaler.inverse_transform(y_test.numpy())
    test_predictions_unscaled = scaler.inverse_transform(test_predictions.numpy())
    
    # --- E. Calculate Prediction Error for Fatigue Detection ---
    print("\n[5/6] Calculating prediction error for fatigue analysis...")
    # The error is the key indicator for fatigue.
    # As the rod fatigues, its vibration changes, and the error of our model
    # (trained on "healthy" data) will increase.
    prediction_error = np.abs(y_test_unscaled - test_predictions_unscaled)

    # --- F. Plot the Results ---
    print("\n[6/6] Plotting results...")
    
    # Plot 1: Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log') # Log scale is often useful for viewing loss
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Actual vs. Predicted on the Test Set
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_unscaled, label='Actual Vibration (ax)', color='blue', alpha=0.7)
    plt.plot(test_predictions_unscaled, label='Predicted Vibration (ax)', color='red', linestyle='--')
    plt.title('Vibration Prediction on Test Data')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration (Original Scale)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 3: Prediction Error (The Fatigue Indicator!)
    plt.figure(figsize=(14, 7))
    plt.plot(prediction_error, label='Prediction Error (Absolute)', color='green')
    plt.title('Prediction Error Over Time (Fatigue Indicator)')
    plt.xlabel('Time Step (in the test set)')
    plt.ylabel('Absolute Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n--- Analysis Complete ---")

