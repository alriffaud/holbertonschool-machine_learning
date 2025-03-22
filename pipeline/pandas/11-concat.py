#!/usr/bin/env python3
"""
This script defines the concat function.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    This function takes two pd.DataFrame objects and performs the following
    operations:
    - Indexes both dataframes on their Timestamp columns.
    - Includes all timestamps from df2 (bitstamp) up to and including timestamp
        1417411920.
    - Concatenates the selected rows from df2 to the top of df1 (coinbase).
    - Adds keys to the concatenated data, labeling the rows from df2 as
        bitstamp and the rows from df1 as coinbase.
    Args:
        df1 (pd.DataFrame): The first input DataFrame.
        df2 (pd.DataFrame): The second input DataFrame.
    Returns:
        The concatenated pd.DataFrame.
    """
    # Set 'Timestamp' as the index for both DataFrames
    index(df1)
    index(df2)

    # Filter df2 to include only timestamps <= 1417411920
    df2_filtered = df2[df2.index <= 1417411920]

    # Concatenate the DataFrames with labeled keys
    df_final = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

    return df_final
