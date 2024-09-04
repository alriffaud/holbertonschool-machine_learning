#!/usr/bin/env python3
""" This module defines the save_weights and load_weights functions. """
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    This function saves a model's weights.
    Args:
        network (model): whose weights should be saved
        filename (str): is the path of the file that the weights should be
        saved to.
        save_format (str): is the format in which the weights should be saved.
    Returns: None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    This function loads a model's weights.
    Args:
        network (model): is the model to which the weights should be loaded.
        filename (str):  is the path of the file that the weights should be
        loaded from.
    Returns: None.
    """
    network.load_weights(filename)
    return None
