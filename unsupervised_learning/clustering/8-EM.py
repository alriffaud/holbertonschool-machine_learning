#!/usr/bin/env python3
""" This module defines the expectation_maximization function. """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    This function performs the expectation maximization for a GMM.
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations for the algorithm.
        tol: non-negative float containing tolerance of the log likelihood,
            used to determine early stopping i.e. if the difference is less
            than or equal to tol, the algorithm should stop.
        verbose: boolean that determines if you should print information about
            the algorithm.
    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.
        g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        l is the log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    # Initialize the GMM
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None
    # Perform the EM algorithm
    l_prev = 0
    for i in range(iterations):
        g, l_log = expectation(X, pi, m, S)
        if g is None or l_log is None:
            return None, None, None, None, None
        # Print verbose output every 10 iterations
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {l_log:.5f}")

        # Check convergence
        if abs(l_log - l_prev) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {l_log:.5f}")
            break
        l_prev = l_log

        # Maximization Step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    # Return final parameters and log likelihood
    return pi, m, S, g, l_log
