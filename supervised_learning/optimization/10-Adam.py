#!/usr/bin/env python3
""" This module defines the create_Adam_op function. """
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    This function sets up the Adam optimization algorithm in TensorFlow.
    Args:
        alpha (float): is the learning rate.
        beta1 (float): is the weight used for the first moment.
        beta2 (float): is the weight used for the second moment.
        epsilon (float): is a small number to avoid division by zero.
    Returns: the Adam optimization operation.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,
                                         beta_1=beta1,
                                         beta_2=beta2,
                                         epsilon=epsilon)
    return optimizer
