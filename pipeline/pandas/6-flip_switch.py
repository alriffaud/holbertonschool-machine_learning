#!/usr/bin/env python3
"""
This script defines the function flip_switch
"""


def flip_switch(df):
    """
    This function takes a pd.DataFrame as input and performs the following
    operations:
    - Sorts the data in reverse chronological order.
    - Transposes the sorted dataframe.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        The transformed DataFrame
    """
    # Sort the DataFrame by the 'Timestamp' column in descending order
    df_sorted = df.sort_values(by='Timestamp', ascending=False)

    # Transpose the sorted DataFrame
    df_transpose = df_sorted.T

    return df_transpose
