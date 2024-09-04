#!/usr/bin/env python3
""" This module defines the save_model and load_model functions. """
import tensorflow.keras as K


def save_model(network, filename):
    """
    This function saves an entire model.
    Args:
        network (model): is the model to save.
        filename (str): is the path of the file that the model should be
        saved to.
    Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    This function loads an entire model.
    Args:
        filename (str):  is the path of the file that the model should be
        loaded from.
    Returns: the loaded model.
    """
    return K.models.load_model(filename)
