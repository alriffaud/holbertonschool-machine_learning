#!/usr/bin/env python3
""" This module defines the predict function. """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    This function makes a prediction using a neural network.
    Args:
        network (model): is the network model to make the prediction with.
        data (numpy.ndarray): is the input data to make the prediction with.
        verbose (bool): determines if output should be printed during the
        prediction process.
    Returns: the prediction for the data.
    """
    return network.predict(data, verbose=verbose)
