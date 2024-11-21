#!/usr/bin/env python3
""" This module defines the expectation function. """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    This function calculates the expectation step in the EM algorithm for a
    GMM.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster.
        m: numpy.ndarray of shape (k, d) containing the centroid means for each
            cluster.
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
            for each cluster.
    Returns:
        g, l or None, None on failure.
        g is numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        l is the total log likelihood.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    # Get the number of data points and dimensions
    n, d = X.shape
    # Get the number of clusters
    k = pi.shape[0]
    # Check if the dimensions of pi, m, and S are correct
    if (d != m.shape[1] or d != S.shape[1]
            or d != S.shape[2] or k != S.shape[0]):
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    # Calculate the likelihood for each data point in each cluster
    likelihoods = np.zeros((k, n))
    for i in range(k):
        likelihoods[i] = pi[i] * pdf(X, m[i], S[i])
    # Calculate the total log likelihood
    lh = np.sum(np.log(np.sum(likelihoods, axis=0)))
    # Calculate the posterior probabilities
    g = likelihoods / np.sum(likelihoods, axis=0)
    return g, lh
