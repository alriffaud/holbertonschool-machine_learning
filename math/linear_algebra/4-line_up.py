#!/usr/bin/env python3
"""This file defines the add_arrays function"""


def add_arrays(arr1, arr2):
    """
    This function adds two arrays element-wise.
    Args:
        arr1 (list): this is the first array to add.
        arr2 (list): this is the second array to add.
    """
    n = len(arr1)
    m = len(arr2)
    if n != m:
        return None
    else:
        return [arr1[i] + arr2[i] for i in range(n)]
