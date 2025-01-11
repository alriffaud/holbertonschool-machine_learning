#!/usr/bin/env python3
"""
Train and validate an RNN model for BTC price forecasting.
"""
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from datetime import datetime, timezone


def load_data(file_path, batch_size=32):
    """
    Load preprocessed data and create a tf.data.Dataset.
    Arguments:
    - file_path: path to the preprocessed data (.npz format).
    - batch_size: batch size for the dataset.
    Returns:
    - train_ds: training dataset.
    - val_ds: validation dataset.
    """
    # Load the preprocessed data
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    timestamps = data["timestamps"]
    mean = data["mean"]
    std = data["std"]

    # Split into training and validation sets
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    timestamps_train = timestamps[:split_idx]
    timestamps_val = timestamps[split_idx:]

    # Create tf.data.Dataset
    train_ds = Dataset.from_tensor_slices(
        (X_train, y_train)).batch(batch_size).shuffle(1000)
    val_ds = Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    return train_ds, val_ds, timestamps_val, mean, std


def create_model(input_shape):
    """
    Create an LSTM-based RNN model.
    Arguments:
    - input_shape: shape of the input data (window_size, 1).
    Returns:
    - model: compiled Keras model.
    """
    # Define the model
    model = Sequential([
        LSTM(64, activation="tanh",
             return_sequences=False, input_shape=input_shape),
        Dense(1)  # Predict a single value (next closing price)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Arguments:
    - y_true: true values.
    - y_pred: predicted values.
    Returns:
    - mape: MAPE value as a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


if __name__ == "__main__":
    # Load data
    train_ds, val_ds, timestp_val, mean, std = load_data("processed_data.npz")

    # Create the model
    model = create_model((24, 1))  # 24 hours, 1 feature

    # Train the model
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Save the model
    model.save("btc_forecast_model.keras")

    # Evaluate the model
    val_X, val_y = next(iter(val_ds.unbatch().batch(len(val_ds))))
    predictions = model.predict(val_X)

    # Denormalize predictions and targets
    predictions = predictions * std + mean
    val_y = val_y * std + mean

    # Calculate MAPE
    mape = calculate_mape(val_y, predictions)
    print(f"MAPE: {mape:.2f}%")

    # Convert timestamps to readable dates
    readable_dates = [
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%d/%m/%Y %H:%M") for ts in timestp_val[:len(val_y)]]

    # Filter indexes and dates that correspond to exact minutes
    filtered_indices = [i for i, ts in enumerate(timestp_val[:len(val_y)])
                        if datetime.fromtimestamp(ts,
                                                  tz=timezone.utc).minute == 0]
    filtered_dates = [readable_dates[i] for i in filtered_indices]

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(readable_dates, val_y, label="Actual Prices", color="blue")
    plt.plot(readable_dates, predictions,
             label="Predicted Prices", color="red")
    plt.title("BTC Price Forecasting - Actual vs Predicted")
    plt.xlabel("Date and Time")
    plt.ylabel("BTC Price (USD)")
    # Set X-axis labels to show only exact minutes
    plt.xticks(filtered_indices, filtered_dates, rotation=45, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the last predicted value
    print(f"Next predicted BTC closing price: {predictions[-1][0]:.2f} USD")
