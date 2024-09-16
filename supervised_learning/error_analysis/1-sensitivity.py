#!/usr/bin/env python3
""" This module defines the sensitivity function. """
import numpy as np


def sensitivity(confusion):
    """
    This function calculates the sensitivity for each class in a confusion
    matrix.
    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
        (classes, classes) with row indices representing the correct labels and
        column indices representing the predicted labels.
    Returns: A numpy.ndarray of shape (classes,) containing the sensitivity of
    each class.
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
