#!/usr/bin/env python3
""" This module defines the create_mini_batches function. """
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    This function creates mini-batches to be used for training a neural network
    using mini-batch gradient descent.
    This function allows for a smaller final batch (i.e. uses the entire
    dataset).
    Args:
        X (numpy.ndarray): matrix of shape (m, nx) to normalize. m is the
        number of data points. nx is the number of features.
        Y (numpy.ndarray): matrix of shape (m, ny) to normalize. m is the
        number of data points. ny is the number of output features.
        batch_size (int): the number of data points in a batch.
    Returns: list of mini-batches containing tuples (X_batch, Y_batch).
    """
    m = X.shape[0]
    mini_batches = []
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    complete_batches = m // batch_size
    for i in range(complete_batches):
        X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    if m % batch_size != 0:
        X_batch = X_shuffled[complete_batches * batch_size:]
        Y_batch = Y_shuffled[complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
