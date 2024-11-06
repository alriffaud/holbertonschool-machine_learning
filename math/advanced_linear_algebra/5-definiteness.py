#!/usr/bin/env python3
""" This module defines the definiteness function wich returns the definiteness
of a matrix. """
import numpy as np


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
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None
    if not all(len(row) == len(matrix) for row in matrix):
        return None
    eigenvalues = np.linalg.eigvals(matrix)
    if np.any(np.iscomplex(eigenvalues)):
        return None
    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues >= 0) and np.any(eigenvalues == 0):
        return "Positive semi-definite"
    if np.all(eigenvalues <= 0) and np.any(eigenvalues == 0):
        return "Negative semi-definite"
    return "Indefinite"
