#!/usr/bin/env python3
""" This module defines the one_hot_decode function. """
import numpy as np


def one_hot_decode(one_hot):
    """
    This function converts a one-hot matrix into a a vector of labels.
    one_hot (numpy.ndarray): is a one-hot matrix with shape (classes, m).
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    return np.argmax(one_hot, axis=0)
