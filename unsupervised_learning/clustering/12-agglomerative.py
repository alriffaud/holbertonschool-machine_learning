#!/usr/bin/env python3
""" This module defines the agglomerative function. """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    This function performs agglomerative clustering on a dataset.
    This function performs agglomerative clustering with Ward linkage.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
            n is the number of data points.
            d is the number of dimensions for each data point.
        dist: maximum cophenetic distance for all clusters.
    Returns:
        clss, a numpy.ndarray of shape (n,) containing the cluster indices for
            each data point.
    """
    if not isinstance(dist, (int, float)) or dist < 0:
        return None
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
