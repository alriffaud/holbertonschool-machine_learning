#!/usr/bin/env python3
""" This module defines the gmm function. """
import sklearn.mixture


def gmm(X, k):
    """
    This function calculates a GMM from a dataset.
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
            n is the number of data points.
            d is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.
    Returns:
        pi, m, S, clss, bic or None, None, None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing the priors for each
            cluster.
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster.
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster.
        clss is a numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
        bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
            value for each cluster size tested.
    """
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
