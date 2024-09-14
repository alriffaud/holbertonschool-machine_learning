#!/usr/bin/env python3
""" This module defines the create_batch_norm_layer function. """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    This function creates a batch normalization layer for a neural network in
    tensorflow.
    Args:
        prev (tensorflow.Tensor): is the activated output of the previous
        layer.
        n (int): is the number of nodes in the layer to be created.
        activation (tensorflow.nn.activation): is the activation function that
        should be used on the output of the layer.
    Returns: a tensor of the activated output for the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init,
                                  name="layer")
    Z = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)
    return activation(Z_norm)
