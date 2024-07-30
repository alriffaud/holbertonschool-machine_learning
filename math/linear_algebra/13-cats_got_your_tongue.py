#!/usr/bin/env python3
"""This file defines the np_cat function"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function concatenates two matrices along a specific axis.
    Args:
        mat1 (list): this is the first matrix.
        mat2 (list): this is the second matrix.
        axis (int): this is the axis used to concatenate two matrices.
    Returns:
        This function returns a matrix representing the concatenation of the
        two matrices.
    """
    return np.concatenate((mat1, mat2), axis)
