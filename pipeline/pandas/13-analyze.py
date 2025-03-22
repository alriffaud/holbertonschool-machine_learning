#!/usr/bin/env python3
"""
This script defines the analyze function.
"""


def analyze(df):
    """
    This function takes a pd.DataFrame as input and computes descriptive
    statistics for all columns except the Timestamp column.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        A new pd.DataFrame containing these statistics.
    """
    df_subset = df.drop(columns=['Timestamp'])
    return df_subset.describe()
