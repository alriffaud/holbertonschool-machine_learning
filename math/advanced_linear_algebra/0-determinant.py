#!/usr/bin/env python3
""" This module calculate the determinant of a matrix. """


def determinant(matrix):
    """
    This function calculate the determinant of a matrix.
    Args:
        matrix (list): A list of lists whose determinant should be calculated.
    Returns:
        int: The determinant of the matrix.
    """
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or matrix == [[]] or len(matrix) == 0:
        return 1
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i, num in enumerate(matrix[0]):
        sub_matrix = [row for row in matrix[1:]]
        for j, row in enumerate(sub_matrix):
            sub_matrix[j] = row[:i] + row[i + 1:]
        det += num * (-1) ** i * determinant(sub_matrix)
    return det
