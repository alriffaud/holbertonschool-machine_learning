#!/usr/bin/env python3
"""This module defines the function forward_prop."""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    This function creates the forward propagation graph for the neural network.
    Args:
        x (tensor): is the placeholder for the input data.
        layer_sizes (list of int): is a list containing the number of nodes in
        each layer of the network.
        activations (list of functions):  is a list containing the activation
        functions for each layer of the network.
    """
    m = len(layer_sizes)
    n = len(activations)
    if m == n:
        for i in range(n):
            x = create_layer(x, layer_sizes[i], activations[i])
    return x
