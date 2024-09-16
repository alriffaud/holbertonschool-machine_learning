#!/usr/bin/env python3
""" This module defines the precision function. """
import numpy as np


def precision(confusion):
    """
    This function calculates the precision for each class in a confusion
    matrix.
    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
        (classes, classes) with row indices representing the correct labels and
        column indices representing the predicted labels.
    Returns: A numpy.ndarray of shape (classes,) containing the precision of
    each class.
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
