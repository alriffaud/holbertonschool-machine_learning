#!/usr/bin/env python3
""" This module defines the test_model function. """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    This function tests a neural network.
    Args:
        network (model): is the network model to test.
        data (numpy.ndarray): is the input data to test the model with.
        labels (numpy.ndarray): is the correct one-hot labels of data.
        verbose (bool): determines if output should be printed during the
        testing process.
    Returns: the loss and accuracy of the model with the testing data,
    respectively.
    """
    # Evaluate the model with the input data and labels
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
