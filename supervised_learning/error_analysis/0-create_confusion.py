#!/usr/bin/env python3
""" This module defines the create_confusion_matrix function. """
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    This function creates a confusion matrix.
    Args:
        labels (numpy.ndarray): A numpy array of shape (m, classes) containing
        the correct labels for each data point. m is the number of data points.
        classes is the number of classes.
        logits (numpy.ndarray): A numpy array of shape (m, classes) containing
        the predicted labels.
    Returns: A confusion numpy.ndarray of shape (classes, classes) with row
    indices representing the correct labels and column indices representing
    the predicted labels.
    """
    return np.matmul(labels.T, logits)
