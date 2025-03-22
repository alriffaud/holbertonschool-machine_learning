#!/usr/bin/env python3
"""
This script defines the fill function.
"""


def fill(df):
    """
    This function takes a pd.DataFrame as input and performs the following
    operations:
    - Removes the Weighted_Price column.
    - Fills missing values in the Close column with the previous row's value.
    - Fills missing values in the High, Low, and Open columns with the
        corresponding Close value in the same row.
    - Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The modified DataFrame
    """
    # Remove the 'Weighted_Price' column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing values in 'Close' with the previous row's value
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing values in 'High', 'Low', 'Open' with their corresponding
    # 'Close' value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Fill missing values in 'Volume_(BTC)' and 'Volume_(Currency)' with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
