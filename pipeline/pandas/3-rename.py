#!/usr/bin/env python3
"""
This script defines the function rename
"""
import pandas as pd


def rename(df):
    """
    takes a pd.DataFrame as input and performs the following operations:
    - The function should rename the Timestamp column to Datetime.
    - Convert the timestamp values to datatime values
    - Display only the Datetime and Close column
    Args:
        df (pd.DataFrame): The input DataFrame to rename
    Returns:
        The modified DataFrame
    """
    # Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the 'Datetime' column from UNIX timestamp to readable datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Select only the 'Datetime' and 'Close' columns
    df = df[['Datetime', 'Close']]

    return df
