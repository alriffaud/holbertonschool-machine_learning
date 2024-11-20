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
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    # Initialize centroids
    C = initialize(X, k)
    # If centroids can't be initialized
    if C is None:
        return None, None
    # Perform K-means iterations
    for _ in range(iterations):
        # Copy the previous centroids
        prev_C = np.copy(C)
        # Calculate distances
        X_vector = np.repeat(X, k, axis=0)
        # Reshape to calculate distances
        X_vector = X_vector.reshape(n, k, d)
        centroids_vector = np.tile(C, (n, 1))
        centroids_vector = centroids_vector.reshape(n, k, d)
        # Calculate distances
        distances = np.linalg.norm(X_vector - centroids_vector, axis=2)
        # Assign data points to closest centroid
        clss = np.argmin(distances ** 2, axis=1)
        # Update centroids
        new_C = np.empty_like(C)
        # If a cluster is empty, reinitialize a random centroid
        for i in range(k):
            indixes = np.where(clss == i)[0]
            if len(indixes) == 0:
                C[i] = initialize(X, 1)
            else:
                C[i] = np.mean(X[indixes], axis=0)
        # If the centroids don't change, stop the iterations
        if np.all(prev_C == C):
            return C, clss
        # If the centroids have not converged, repeat the process
        centroids_vector = np.tile(C, (n, 1))
        centroids_vector = centroids_vector.reshape(n, k, d)
        distance = np.linalg.norm(X_vector - centroids_vector, axis=2)
        clss = np.argmin(distance ** 2, axis=1)
    return C, clss
