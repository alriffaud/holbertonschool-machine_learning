#!/usr/bin/env python3
""" This module defines the create_momentum_op function. """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    This function sets up the gradient descent with momentum optimization
    algorithm in TensorFlow.
    Args:
        alpha (float): is the learning rate.
        beta1 (float): is the momentum weight.
    Returns: the momentum optimization operation.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
