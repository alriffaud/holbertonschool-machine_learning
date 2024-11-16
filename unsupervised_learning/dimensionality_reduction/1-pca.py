#!/usr/bin/env python3
""" This module defines a function that performs PCA on a dataset """
import numpy as np


def pca(X, ndim):
    """
    This function performs PCA on a dataset.
    Args:
        X: numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        ndim: the new dimensionality of the transformed X.
    Returns:
        T, a numpy.ndarray of shape (n, ndim) containing the transformed
        version of X.
    """
    # Subtract the mean of each feature (center the data)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Select the top 'ndim' components
    W = Vt[:ndim].T  # Transpose to get the correct shape

    # Transform the data using 'W'
    T = np.dot(X_centered, W)

    return T
