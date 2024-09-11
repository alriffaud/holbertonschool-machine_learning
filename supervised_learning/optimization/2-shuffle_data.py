#!/usr/bin/env python3
""" This module defines the shuffle_data function. """
import numpy as np


def shuffle_data(X, Y):
    """
    This function shuffles the data points in two matrices the same way.
    Args:
        X (numpy.ndarray): matrix of shape (m, nx) to normalize. m is the
        number of data points. nx is the number of features.
        Y (numpy.ndarray): matrix of shape (m, ny) to normalize. m is the
        number of data points. ny is the number of output features.
        Returns: the shuffled X and Y matrices.
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
