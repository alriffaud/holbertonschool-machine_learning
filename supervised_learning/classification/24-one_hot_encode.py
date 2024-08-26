#!/usr/bin/env python3
""" This module defines the one_hot_encode function. """
import numpy as np


def one_hot_encode(Y, classes):
    """
    This function converts a numeric label vector into a one-hot matrix.
    Y (numpy.ndarray): is a numeric label vector with shape (m,).
    classes (int): is the maximum number of classes found in Y.
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, len(Y)))
    one_hot[Y, np.arange(len(Y))] = 1
    return one_hot
