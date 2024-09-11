#!/usr/bin/env python3
""" This module defines the normalization_constants function. """
import numpy as np


def normalization_constants(X):
    """
    This function calculates the normalization (standardization) constants of
    a matrix.
    Args:
        X (numpy.ndarray): matrix of shape (m, nx) to normalize. m is the
        number of data points. nx is the number of features.
    Returns: the mean and standard deviation of each feature, respectively.
    """
    mean_list = X.mean(axis=0)
    std_list = X.std(axis=0)
    return mean_list, std_list
