#!/usr/bin/env python3
""" This module defines the l2_reg_create_layer function. """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ This function creates a neural network layer in tensorFlow that
    includes L2 regularization.
    Args:
        prev (tf.Tensor): is a tensor containing the output of the previous
        layer.
        n (int): is the number of nodes the new layer should contain.
        activation (function): is the activation function that should be used
        on the layer.
        lambtha (float): is the L2 regularization parameter.
    Returns: the output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    l2_reg = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init,
                                  kernel_regularizer=l2_reg)
    return layer(prev)
