#!/usr/bin/env python3
"""This file defines the np_slice function"""


def np_slice(matrix, axes={}):
    """
    This function slices a matrix along specific axes.
    Args:
        matrix (numpy.ndarray): this is the matrix to slice.
        axes: a dictionary where the key is an axis to slice along and the
        value is a tuple representing the slice to make along that axis.
    Returns:
        This function returns a matrix representing a matrix along specific
        axes.
    """
    slices = [slice(None)] * matrix.ndim
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
