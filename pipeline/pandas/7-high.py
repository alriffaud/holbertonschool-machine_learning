#!/usr/bin/env python3
"""
This script defines the high function.
"""


def high(df):
    """
    This function takes a pd.DataFrame as input and performs the following
    operations:
    - Sorts it by the High price in descending order.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The sorted DataFrame
    """
    # Sort the DataFrame by the 'High' column in descending order
    df_sorted = df.sort_values(by='High', ascending=False)

    return df_sorted
