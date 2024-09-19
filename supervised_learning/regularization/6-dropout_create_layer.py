#!/usr/bin/env python3
""" This module defines the dropout_create_layer function. """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """ This function creates a neural network layer in tensorFlow using
    dropout.
    Args:
        prev (tf.Tensor): is a tensor containing the output of the previous
        layer.
        n (int): is the number of nodes the new layer should contain.
        activation (function): is the activation function for the new layer.
        keep_prob (float): is the probability that a node will be kept.
        training (bool): is a boolean that determines if the model is in
        training mode.
    Returns: the output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init, name='layer')
    drop = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return drop(layer(prev), training=training)
