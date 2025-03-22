#!/usr/bin/env python3
"""
This script defines the function array
"""


def array(df):
    """
    This function takes a pd.DataFrame as input and performs the following
    operations:
    - The function should select the last 10 rows of the High and Close columns
    - Convert these selected values into a numpy.ndarray
    Args:
        df (pd.DataFrame): The input DataFrame to select the last 10 rows
    Returns:
        The newly created numpy.ndarray
    """
    # Select only the 'Datetime' and 'Close' columns
    df_subset = df[['High', 'Close']]

    # Select the last 10 rows
    df_subset = df_subset.tail(10)

    # Convert the DataFrame to a numpy array
    return df_subset.to_numpy()
