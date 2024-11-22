#!/usr/bin/env python3
""" This module defines the kmeans function. """
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """
    This function performs K-means on a dataset.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
            n is the number of data points.
            d is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.
    Returns:
        C, clss or None, None on failure.
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
