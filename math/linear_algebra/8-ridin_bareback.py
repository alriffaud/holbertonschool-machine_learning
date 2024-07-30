#!/usr/bin/env python3
"""This file defines the mat_mul function"""


def dot_prod(arr1, arr2):
    """
    This function obtains the dot product of two arrays.
    Args:
        arr1 (list): this is the first array to multiply.
        arr2 (list): this is the second array to multiply.
    Returns:
        This function returns the dot product. None otherwise.
    """
    if not arr1 or not arr2 or len(arr1) != len(arr2):
        return None
    dot_prod = arr1[0] * arr2[0]
    for i in range(1, len(arr1)):
        dot_prod += arr1[i] * arr2[i]
    return dot_prod


def mat_mul(mat1, mat2):
    """
    This function performs matrix multiplication.
    Args:
        mat1 (list): this is the first matrix to multiply.
        mat2 (list): this is the second matrix to multiply.
    Returns:
        This function returns the product of the two matrix.
        None otherwise.
    """
    if not mat1 or not mat2 or len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            col_j = [mat2[k][j] for k in range(len(mat2))]
            row.append(dot_prod(mat1[i], col_j))
        result.append(row)
    return result
