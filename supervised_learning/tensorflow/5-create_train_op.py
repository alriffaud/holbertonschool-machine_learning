#!/usr/bin/env python3
"""This module defines the function create_train_op."""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    This function creates the training operation for the network.
    Args:
        loss (tensor): is the loss of the network's prediction.
        alpha (float): is the learning rate.
    Returns: an operation that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
