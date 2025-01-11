# BTC Forecasting with RNNs

## Overview
This project uses an RNN-based architecture to forecast the closing price of Bitcoin (BTC) using historical data.

## Files
- `preprocess_data.py`: Preprocesses raw BTC datasets for time series forecasting.
- `forecast_btc.py`: Creates, trains, and validates the RNN model.
- `processed_data.npz`: Contains preprocessed data for training.

## Requirements
- Python 3.9
- TensorFlow 2.15
- Numpy 1.25.2
- Pandas 2.2.2

## Usage
1. Preprocess the data:
   ```bash
   ./preprocess_data.py
2. Train the model:
    ```bash
    ./forecast_btc.py
