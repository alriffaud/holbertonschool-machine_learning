#!/usr/bin/env python3
""" This module defines the correlation function. """
import numpy as np


def correlation(C):
    """
    This function calculates the correlation matrix of a data set.
    Args:
        C (numpy.ndarray): Is an array of shape (d, d) containing a covariance
            matrix. d is the number of dimensions.
    Returns:
        An array of shape (d, d) containing the correlation matrix of the data
            set.
    """
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    D = np.diag(1 / np.sqrt(np.diag(C)))
    correlation = np.dot(np.dot(D, C), D)
    return correlation
