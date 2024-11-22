#!/usr/bin/env python3
""" This module defines the BIC function. """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    This function finds the best number of clusters for a GMM using the
    Bayesian Information Criterion.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        kmin: positive integer containing the minimum number of clusters to
            check for.
        kmax: positive integer containing the maximum number of clusters to
            check for.
        iterations: positive integer containing the maximum number of
            iterations for the algorithm.
        tol: is a non-negative float containing the tolerance for the EM
            algorithm
        verbose: boolean that determines if you should print information about
            the algorithm.
    Returns:
        best_k, best_result, l, b, or None, None, None, None on failure.
        best_k is the best value for k based on its BIC.
        best_result is tuple containing pi, m, S, g, l.
            pi is a numpy.ndarray of shape (k,) containing the priors for each
                cluster.
            m is a numpy.ndarray of shape (k, d) containing the centroid means
                for each cluster.
            S is a numpy.ndarray of shape (k, d, d) containing the covariance
                matrices for each cluster.
            g is a numpy.ndarray of shape (k, n) containing the posterior
                probabilities for each data point in each cluster.
            l is the log likelihood of the model.
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested.
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int)
                             or kmax <= 0 or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n  # Set to maximum possible clusters if not specified

    log_likelihoods = []
    bics = []
    best_k = None
    best_result = None
    best_bic = float('inf')

    for k in range(kmin, kmax + 1):
        # Perform EM algorithm to current number of clusters
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if (pi is None or m is None or S is None or g is None
                or log_likelihood is None):
            return None, None, None, None

        # Compute the number of parameters p
        p = k * d + k * d * (d + 1) // 2 + k - 1

        # Compute BIC
        bic = p * np.log(n) - 2 * log_likelihood

        # Append results
        log_likelihoods.append(log_likelihood)
        bics.append(bic)

        # Check if this is the best BIC
        if bic < best_bic:
            best_k = k
            best_result = (pi, m, S)
            best_bic = bic

    # Convert lists to numpy arrays
    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)

    return best_k, best_result, log_likelihoods, bics
