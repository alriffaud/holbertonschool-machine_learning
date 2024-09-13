#!/usr/bin/env python3
""" This module defines the create_RMSProp_op function. """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    This function sets up the RMSProp optimization algorithm in TensorFlow.
    Args:
        alpha (float): is the learning rate.
        beta2 (float): is the RMSProp weight.
        epsilon (float): is a small number to avoid division by zero.
    Returns: the RMSProp optimization operation.
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)
    return optimizer
