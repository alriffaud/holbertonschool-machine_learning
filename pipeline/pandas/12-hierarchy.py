#!/usr/bin/env python3
"""
This script defines the hierarchy function.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    This function takes two pd.DataFrame objects and performs the following
    operations:
    - Rearranges the MultiIndex so that Timestamp is the first level.
    - Concatenates the bitstamp and coinbase tables from timestamps 1417411980
        to 1417417980, inclusive.
    - Adds keys to the data, labeling rows from df2 as bitstamp and rows from
        df1 as coinbase.
    - Ensures the data is displayed in chronological order.
    Args:
        df1 (pd.DataFrame): The first input DataFrame.
        df2 (pd.DataFrame): The second input DataFrame.
    Returns:
        The concatenated pd.DataFrame.
    """
    # Set 'Timestamp' as the index for both DataFrames
    index(df1)
    index(df2)

    # Concatenate the DataFrames with labeled keys
    df_final = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Swap the MultiIndex levels so that 'Timestamp' is the first level
    df_final = df_final.swaplevel(0, 1)

    # Sort the DataFrame by the MultiIndex to ensure it is ordered by Timestamp
    df_final = df_final.sort_index()

    # Filter timestamps in the range [1417411980, 1417417980]
    df_filtered = df_final.loc[1417411980:1417417980]

    return df_filtered
