#!/usr/bin/env python3
""" This module preprocess raw BTC datasets for time series forecasting. """
import pandas as pd
import numpy as np


def preprocess_data(file_path, output_path, window_size=24):
    """
    Preprocesses raw BTC data for time series forecasting.
    Arguments:
    - file_path: path to the raw dataset (CSV format).
    - output_path: path to save the preprocessed dataset.
    - window_size: size of the sliding window (in hours).
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Select relevant features
    selected_features = ["Timestamp", "Close"]
    data = data[selected_features]

    # Handle missing data
    data.dropna(inplace=True)

    # Normalize the data
    mean = data["Close"].mean()
    std = data["Close"].std()
    data["Norm_Close"] = (data["Close"] - mean) / std

    # Create sliding windows
    sequences = []
    targets = []
    timestamps = []
    for i in range(len(data) - window_size - 1):
        sequences.append(data["Norm_Close"].iloc[i:i + window_size].values)
        targets.append(data["Norm_Close"].iloc[i + window_size])
        timestamps.append(data["Timestamp"].iloc[i + window_size])

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    timestamps = np.array(timestamps)

    # Save the processed data
    np.savez(output_path, X=X, y=y, timestamps=timestamps, mean=mean, std=std)


if __name__ == "__main__":
    # Example usage
    preprocess_data("coinbaseUSD_1-min.csv", "processed_data.npz")
    print("Data preprocessing complete. File saved as processed_data.npz")
