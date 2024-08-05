#!/usr/bin/env python3
"""This file defines a new function"""


def matrix_shape(matrix):
    """
    This function calculates the shape of a matrix.
    Args:
        matrix: matrix to calculate its shape.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if not matrix[0]:
            return shape
        matrix = matrix[0]
    return shape
