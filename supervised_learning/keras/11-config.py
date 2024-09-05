#!/usr/bin/env python3
""" This module defines the save_config and load_config functions. """
import tensorflow.keras as K


def save_config(network, filename):
    """
    This function saves a model's configuration in JSON format.
    Args:
        network (model): is the model whose configuration should be saved.
        filename (str): is the path of the file that the configuration should
        be saved to.
    Returns: None
    """
    # Gets the model configuration in JSON format
    config = network.to_json()
    # Open the file in write mode (w) and save the JSON content
    with open(filename, 'w') as json_file:
        json_file.write(config)


def load_config(filename):
    """
    This function loads a model with a specific configuration.
    Args:
        filename (str): is the path of the file containing the model's
        configuration in JSON format.
        loaded from.
    Returns: the loaded model.
    """
    # Read JSON configuration from file
    with open(filename, 'r') as json_file:
        config = json_file.read()
    # Rebuild the model from the JSON configuration
    model = K.models.model_from_json(config)
    return model
