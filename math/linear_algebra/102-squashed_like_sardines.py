#!/usr/bin/env python3
"""This file defines the cat_matrices function"""


def cat_matrices(mat1, mat2, axis=0):
    """
    This function concatenates two matrices along a specific axis.
    Args:
        mat1 (list): this is the first matrix to concatenate.
        mat2 (list): this is the second matrix to concatenate.
    Returns:
        This function returns a new matrix. If the two matrices cannot be
        concatenated, return None.
    """
    if not mat1 or not mat2:
        return None
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # Recursively check the dimensions of the matrices
    def is_list_of_lists(matrix):
        return all(isinstance(elem, list) for elem in matrix)

    def are_shapes_compatible(matrix1, matrix2, depth=0):
        if depth == axis:
            return is_list_of_lists(matrix1) == is_list_of_lists(matrix2)
        if not is_list_of_lists(matrix1) or not is_list_of_lists(matrix2):
            return False
        if len(matrix1) != len(matrix2):
            return False
        return (all(are_shapes_compatible(matrix1[i], matrix2[i], depth+1)
                    for i in range(len(matrix1))))

    if not are_shapes_compatible(mat1, mat2):
        return None

    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    elif axis == 2:
        if (any(len(row) != len(mat1[0]) for row in mat1)
                or any(len(row) != len(mat1[0]) for row in mat2)):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    elif axis == 3:
        if len(mat1) != len(mat2):
            return None
        return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
                for i in range(len(mat1))]
    return None
