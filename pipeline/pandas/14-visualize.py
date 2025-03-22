#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove 'Weighted_Price' column
df.drop(columns=['Weighted_Price'], inplace=True)

# Rename 'Timestamp' column to 'Date'
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Fill missing 'Close' values with previous row value
df["Close"] = df["Close"].ffill()

# Fill missing values in 'High', 'Low', 'Open' with their corresponding
# 'Close' value
df['High'] = df['High'].fillna(df['Close'], axis=0)
df['Low'] = df['Low'].fillna(df['Close'], axis=0)
df['Open'] = df['Open'].fillna(df['Close'], axis=0)

# Fill missing values in 'Volume_(BTC)' and 'Volume_(Currency)' with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 onwards
df = df[df.index >= '2017-01-01']

# Resample data to daily intervals and apply required aggregations
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print transformed DataFrame before plotting
print(df)

# Plot the closing price over time
labels = ["High", "Low", "Open", "Close", "Volume_(BTC)", "Volume_(Currency)"]
df[labels].plot(figsize=(8, 6))
plt.title("Daily Bitcoin Prices (2017 and Beyond)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(labels)
plt.show()
