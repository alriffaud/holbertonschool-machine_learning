#!/usr/bin/env python3
""" This module defines the kmeans function. """
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


def kmeans(X, k, iterations=1000):
    """
    This function performs K-means on a dataset.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
           used for K-means clustering.
            n is the number of data points.
            d is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.
        iterations: is a positive integer containing the maximum number of
                    iterations that should be performed.
    Returns:
        C, clss, or None, None on failure. C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster. clss is a numpy.ndarray
        of shape (n,) containing the index of the cluster in C that each data
        point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(iterations, int) or iterations <= 0:
        return None
    n, d = X.shape
    # Initialize centroids
    C = initialize(X, k)
    # If centroids can't be initialized
    if C is None:
        return None, None
    # Perform K-means iterations
    for _ in range(iterations):
        # Assign data points to closest centroid
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        # If no change in the cluster centroids
        new_C = np.copy(C)
        # Update centroids
        for i in range(k):
            # If there are no data points for a cluster, reinitialize
            if i not in clss:
                new_C[i] = np.random.uniform(np.min(X, axis=0),
                                             np.max(X, axis=0), (1, d))
            else:
                # Update centroid
                new_C[i] = np.mean(X[clss == i], axis=0)
        # If no change in the cluster centroids
        if np.all(new_C == C):
            return C, clss
        # Update centroids
        C = new_C
    # Assign data points to closest centroid
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return C, clss
