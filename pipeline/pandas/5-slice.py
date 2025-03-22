#!/usr/bin/env python3
"""
This script defines the function slice
"""


def slice(df):
    """
    This function takes a pd.DataFrame as input and performs the following
    operations:
    - Extracts the columns High, Low, Close, and Volume_BTC
    - Selects every 60th row from these columns
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The sliced pd.DataFrame
    """
    # Select only the High, Low, Close, and Volume_(BTC) columns
    df_subset = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    # Select every 60th row using slicing
    df_sliced = df_subset.iloc[::60]

    return df_sliced
