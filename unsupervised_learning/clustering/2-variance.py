#!/usr/bin/env python3
""" This module defines the variance function. """
import numpy as np


def variance(X, C):
    """
    This function calculates the total intra-cluster variance for a data set.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        C: numpy.ndarray of shape (k, d) containing the centroid means for each
           cluster.
            k is the number of clusters.
    Returns:
        var, or None on failure.
        var is the total variance.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    n, d = X.shape
    # Check if C is a valid cluster
    k = C.shape[0]
    if d != C.shape[1]:
        return None
    if k > n:
        return None
    if k <= 0:
        return None
    # Calculate the squared Euclidean distance between each data point and
    # the centroid of its cluster
    dist = np.sqrt(np.sum((X[:, np.newaxis] - C) ** 2, axis=-1))
    # Calculate the variance
    min_dist = np.min(dist, axis=-1)
    return np.sum(min_dist ** 2)
