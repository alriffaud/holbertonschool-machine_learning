#!/usr/bin/env python3
""" This module defines the f1_score function. """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    This function calculates the f1_score for each class in a confusion
    matrix.
    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
        (classes, classes) with row indices representing the correct labels and
        column indices representing the predicted labels.
    Returns: A numpy.ndarray of shape (classes,) containing the f1_score of
    each class.
    """
    sen_arr = sensitivity(confusion)
    pre_arr = precision(confusion)
    F1_sc = 2 * (sen_arr * pre_arr) / (sen_arr + pre_arr)
    return F1_sc
