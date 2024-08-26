#!/usr/bin/env python3
""" This module defines the one_hot_encode function. """
import numpy as np


def one_hot_encode(Y, classes):
    """
    This function converts a numeric label vector into a one-hot matrix.
    Y (numpy.ndarray): is a numeric label vector with shape (m,).
    classes (int): is the maximum number of classes found in Y.
    """
    if not isinstance(classes, int) or classes < 1:
        return None
    one_hot_matrix = np.zeros((classes, len(Y)))
    for i in range(len(Y)):
        if i >= classes or Y[i] >= len(Y):
            return None
        one_hot_matrix[i][Y[i]] = 1
    return one_hot_matrix.T
