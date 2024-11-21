#!/usr/bin/env python3
""" This module defines the pdf function. """
import numpy as np


def pdf(X, m, S):
    """
    This function calculates the probability density function of a Gaussian
    distribution.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated.
            n is the number of data points.
            d is the number of dimensions in each data point.
        m: numpy.ndarray of shape (d,) containing the mean of the distribution.
        S: numpy.ndarray of shape (d, d) containing the covariance of the
        distribution.
    Returns:
        P, or None on failure.
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
        data point.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or d != S.shape[0] or S.shape[0] != S.shape[1]:
        return None
    # Calculate the determinant of S
    det = np.linalg.det(S)
    if det == 0:
        return None
    # Calculate the inverse of S
    inv = np.linalg.inv(S)
    # Calculate the constant
    C = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    # Calculate the PDF
    P = C * np.exp(-0.5 * np.sum((X - m) @ inv * (X - m), axis=1))
    return P
