#!/usr/bin/env python3
""" This module defines the specificity function. """
import numpy as np


def specificity(confusion):
    """
    This function calculates the specificity for each class in a confusion
    matrix.
    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
        (classes, classes) with row indices representing the correct labels and
        column indices representing the predicted labels.
    Returns: A numpy.ndarray of shape (classes,) containing the specificity of
    each class.
    """
    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, axis=0) - true_positive
    false_negative = np.sum(confusion, axis=1) - true_positive
    true_negative = np.sum(confusion) - (true_positive + false_positive
                                         + false_negative)
    return true_negative / (true_negative + false_positive)
