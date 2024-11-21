#!/usr/bin/env python3
""" This module defines the optimum function. """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    This function tests for the optimum number of clusters by variance.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        kmin: positive integer containing the minimum number of clusters to
        test for (inclusive).
        kmax: positive integer containing the maximum number of clusters to
        test for (inclusive).
        iterations: positive integer containing the maximum number of
        iterations for K-means.
    Returns:
        results, d_vars, or None, None on failure.
        results is a list containing the outputs of K-means for each cluster
        size.
        d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    # Get the number of data points and dimensions
    n, d = X.shape
    # Set kmax to n if not provided
    if kmax is None:
        kmax = n
    if kmin >= kmax:
        return None, None
    # Initialize results and d_vars
    results = []
    variances = []
    # Test for each cluster size
    for k in range(kmin, kmax + 1):
        # Perform K-means
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        # Append the results
        results.append((C, clss))
        # Calculate the difference in variance
        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)
    min_variance = variances[0]
    d_vars = [min_variance - var for var in variances]

    return results, d_vars
