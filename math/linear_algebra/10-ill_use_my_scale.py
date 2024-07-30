#!/usr/bin/env python3
"""This file defines the np_shape function"""
import numpy as np


def np_shape(matrix):
    """
    This function calculates the shape of a numpy.ndarray.
    Args:
        matrix: this is the matrix to calculate its shape.
    Returns:
        This function returns a tuple of integers representing the shape
        of the matrix.
    """
    return np.shape(matrix)
