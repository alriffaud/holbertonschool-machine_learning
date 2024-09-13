#!/usr/bin/env python3
""" This module defines the learning_rate_decay function. """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    This function creates a learning rate decay operation in tensorflow using
    inverse time decay.
    Args:
        alpha (float): is the original learning rate.
        decay_rate (float): is the weight used to determine the rate at which
        alpha will decay.
        decay_step (int): is the number of passes of gradient descent that
        should occur before alpha is decayed further.
    Returns: the updated value for alpha.
    """
    operation = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
    return operation
