#!/usr/bin/env python3
""" This module defines the mean_cov function. """
import numpy as np


def mean_cov(X):
    """
    This function calculates the mean and covariance of a data set.
    Args:
        X (numpy.ndarray): Is an array of shape (n, d) containing the data set.
    Returns:
        The mean and covariance of the data set.
        mean (numpy.ndarray): Is an array of shape (1, d) containing the mean
            of the data set.
        cov (numpy.ndarray): Is an array of shape (d, d) containing the
            covariance matrix of the data set.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.dot((X - mean).T, X - mean) / (n - 1)
    return mean, cov
