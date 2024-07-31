#!/usr/bin/env python3
"""This file defines the add_matrices function"""


def add_matrices(mat1, mat2):
    """
    This function adds two matrices.
    Args:
        mat1 (list): this is the first matrix.
        mat2 (list): this is the second matrix.
    Returns:
        This function returns a matrix representing the sum of the
        two matrices.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if len(mat1) != len(mat2):
        return None

    if not isinstance(mat1[0], list):  # 1D matrix
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    new_matrix = []
    for i in range(len(mat1)):
        if not isinstance(mat1[i], list) or not isinstance(mat2[i], list):
            return None

        if len(mat1[i]) != len(mat2[i]):
            return None

        if isinstance(mat1[i][0], list):  # 3D or 4D matrix
            new_row = []
            for j in range(len(mat1[i])):
                if len(mat1[i][j]) != len(mat2[i][j]):
                    return None

                if not isinstance(mat1[i][j][0], list):  # 4D matrix
                    new_col = []
                    for k in range(len(mat1[i][j])):
                        if len(mat1[i][j][k]) != len(mat2[i][j][k]):
                            return None
                        new_col.append([mat1[i][j][k][t] + mat2[i][j][k][t]
                                        for t in range(len(mat1[i][j][k]))])
                    new_row.append(new_col)
                else:  # 3D matrix
                    new_row.append([mat1[i][j][k] + mat2[i][j][k]
                                    for k in range(len(mat1[i][j]))])
            new_matrix.append(new_row)
        else:  # 2D matrix
            new_matrix.append([mat1[i][j] + mat2[i][j]
                               for j in range(len(mat1[i]))])

    return new_matrix
