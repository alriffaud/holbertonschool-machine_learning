#!/usr/bin/env python3
""" This module defines the one_hot function. """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    This function converts a label vector into a one-hot matrix.
    Args:
        labels (numpy.ndarra): is a list of labels to convert.
        classes (int): is the maximum number of classes found in labels.
        Defaults to None.
    Returns: the one-hot matrix
    """
    # If classes is not provided, we determine it from the maximum value
    # in labels
    if classes is None:
        classes = max(labels) + 1
    # Converts the labels vector into a one-hot matrix
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
