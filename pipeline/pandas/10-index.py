#!/usr/bin/env python3
"""
This script defines the index function.
"""


def index(df):
    """
    This function takes a pd.DataFrame as input and sets the Timestamp
    column as the index of the dataframe.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The modified DataFrame
    """
    return df.set_index('Timestamp')
