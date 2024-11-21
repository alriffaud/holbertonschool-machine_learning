#!/usr/bin/env python3
""" This module defines the maximization function. """
import numpy as np


def maximization(X, g):
    """
    This function calculates the maximization step in the EM algorithm for a
    GMM.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
            for each data point in each cluster.
    Returns:
        pi, m, S or None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing the updated priors for
            each cluster.
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster.
        S is a numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    # Get the number of data points and dimensions
    n, d = X.shape
    # Get the number of clusters
    k = g.shape[0]
    # Verify the shapes
    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None
    # Calculate the updated priors
    pi = np.sum(g, axis=1) / n
    # Calculate the updated means
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    # Calculate the updated covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])
    return pi, m, S
