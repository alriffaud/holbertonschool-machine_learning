#!/usr/bin/env python3
""" This module defines the l2_reg_cost function. """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    This function calculates the cost of a neural network with L2
    regularization.
    Args:
        cost (numpy.ndarray): is a tensor containing the cost of the network
        without L2 regularization.
        model (tf.keras.Model): is a Keras model that includes layers with L2
        regularization.
    Returns: a tensor containing the total cost for each layer of the network,
    accounting for L2 regularization.
    """
    return cost + tf.convert_to_tensor(model.losses)
