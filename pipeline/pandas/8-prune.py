#!/usr/bin/env python3
"""
This script defines the prune function.
"""


def prune(df):
    """
    This function takes a pd.DataFrame as input and removes any entries
    where Close has NaN values.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The modified DataFrame
    """
    # Drop rows where 'Close' is NaN
    df_complete = df.dropna(subset=['Close'])

    return df_complete
