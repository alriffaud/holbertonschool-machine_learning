#!/usr/bin/env python3
"""This file defines the add_matrices2D function"""


def add_matrices2D(mat1, mat2):
    """
    This function adds two matrices element-wise.
    Args:
        mat1 (list): this is the first matrix to add.
        mat2 (list): this is the second matrix to add.
    """
    if len(mat1) != len(mat2):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        else:
            new_row = [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            new_matrix.append(new_row)
    return new_matrix
