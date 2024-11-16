#!/usr/bin/env python3
""" This module defines a function that performs PCA on a dataset """
import numpy as np


def pca(X, var=0.95):
    """
    This function performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        var: fraction of the variance that the PCA transformation should
            maintain.
    Returns:
        The weights matrix, W, that maintains var fraction of X's original
        variance. W is a numpy.ndarray of shape (d, nd) where nd is the new
        dimensionality of the transformed X.
    """
    u, s, vh = np.linalg.svd(X)
    cum_sum = np.cumsum(s)
    threshold = cum_sum[-1] * var
    mask = np.where(cum_sum < threshold, 1, 0)
    r = np.count_nonzero(mask)
    W = vh[:r + 1].T
    return W
