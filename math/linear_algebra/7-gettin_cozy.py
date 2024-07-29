#!/usr/bin/env python3
"""This file defines the cat_matrices2D function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (list): this is the first matrix to concatenate.
        mat2 (list): this is the second matrix to concatenate.
        axis (int, optional): this is the axis to concatenate. Defaults to 0.

    Returns:
        This function returns a new matrix. If the two matrices cannot be
        concatenated, return None.
    """
    if not mat1 or not mat2:
        return None
    if axis == 0:
        n = len(mat1[0])
        for i in range(len(mat1)):
            if len(mat1[i]) != n:
                return None
        for i in range(len(mat2)):
            if len(mat2[i]) != n:
                return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
