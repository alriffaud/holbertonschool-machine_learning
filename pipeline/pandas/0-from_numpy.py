#!/usr/bin/env python3
"""
This script defines a function that creates a DataFrame from a numpy array
"""
import pandas as pd


def from_numpy(array):
    """
    This function creates a DataFrame from a numpy array
    Args:
        array: numpy array from which to create the DataFrame
    Returns:
        The newly created DataFrame
    """
    # Generate column names from 'A' to 'Z' using ASCII values
    column_names = [chr(65 + i) for i in range(array.shape[1])]

    return pd.DataFrame(array, columns=column_names)
