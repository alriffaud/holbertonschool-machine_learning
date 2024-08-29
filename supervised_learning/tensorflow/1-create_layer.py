#!/usr/bin/env python3
"""This module defines the function create_layer."""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    This function creates a layer of a neural network.
    prev (tensor): the tensor output of the previous layer.
    n (int): the number of nodes in the layer to create.
    activation (tf.nn.activation): the activation function that the layer
    should use.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name='layer')
    return layer(prev)
