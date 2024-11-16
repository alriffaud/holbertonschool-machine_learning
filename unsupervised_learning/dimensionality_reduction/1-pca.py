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

    # Calculate the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvectors by eigenvalue in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'ndim' eigenvectors
    W = sorted_eigenvectors[:, :ndim]

    # Transform the data using 'W'
    T = np.dot(X_centered, W)

    return T
