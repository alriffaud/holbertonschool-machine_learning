#!/usr/bin/env python3
"""This file defines a new function"""


def matrix_transpose(matrix):
    """
    This function returns the transpose of a 2D matrix.
    Args:
        matrix (list): matrix to be transposed.
    """
    t_matrix = [[matrix[j][i] for j in range(len(matrix))]
                for i in range(len(matrix[0]))]
    return t_matrix
