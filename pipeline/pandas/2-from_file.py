#!/usr/bin/env python3
"""
This script defines a function that that loads data from a file as a
pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    This function loads data from a file as a pd.DataFrame
    Args:
        filename: name of the file to load
        delimiter: delimiter used in the file
    Returns:
        The newly created DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
