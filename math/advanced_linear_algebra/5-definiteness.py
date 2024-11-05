#!/usr/bin/env python3
""" This module defines the definiteness function wich returns the definiteness
of a matrix. """
import numpy as np
from numpy.linalg import eigvals


def definiteness(matrix):
    """
    This function calculates the definiteness of a matrix.
    Args:
        matrix (numpy.ndarray): is an array of shape (n, n) whose definiteness
            should be calculated.
    Returns:
        str: The definiteness of the matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix == [[]] or len(matrix) == 0:
        return None
    if not all(len(row) == len(matrix) for row in matrix):
        return None
    if not all(isinstance(num, (int, float)) for row in matrix for num in row):
        return None
    if not all(eigvals(matrix)):
        return None
    if all(x > 0 for x in eigvals(matrix)):
        return "Positive definite"
    if all(x < 0 for x in eigvals(matrix)):
        return "Negative definite"
    if all(x >= 0 for x in eigvals(matrix)):
        return "Positive semi-definite"
    if all(x <= 0 for x in eigvals(matrix)):
        return "Negative semi-definite"
    return "Indefinite"
