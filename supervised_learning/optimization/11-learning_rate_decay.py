#!/usr/bin/env python3
""" This module defines the learning_rate_decay function. """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    This function updates the learning rate using inverse time decay in numpy.
    Args:
        alpha (float): is the original learning rate.
        decay_rate (float): is the weight used to determine the rate at which
        alpha will decay.
        global_step (int): is the number of passes of gradient descent that
        have elapsed.
        decay_step (int): is the number of passes of gradient descent that
        should occur before alpha is decayed further.
    Returns: the updated value for alpha.
    """
    alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))
    return alpha
