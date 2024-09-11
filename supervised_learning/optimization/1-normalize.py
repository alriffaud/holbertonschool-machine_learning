#!/usr/bin/env python3
""" This module defines the normalize function. """
import numpy as np


def normalize(X, m, s):
    """
    This function normalizes (standardizes) a matrix.
    Args:
        X (numpy.ndarray): matrix of shape (d, nx) to normalize. d is the
        number of data points. nx is the number of features.
        m (numpy.ndarray): array of shape (nx,) that contains the mean of all
        features of X.
        s (numpy.ndarray): array of shape (nx,) that contains the standard
        deviation of all features of X.
    Returns: The normalized X matrix
    """
    return (X - m) / s
