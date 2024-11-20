#!/usr/bin/env python3
""" This module defines the initialize function. """
import numpy as np


def initialize(X, k):
    """
    This function initializes cluster centroids for K-means.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
           used for K-means clustering.
            n is the number of data points.
            d is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.
    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids for
        each cluster, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))
