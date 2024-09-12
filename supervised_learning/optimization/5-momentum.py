#!/usr/bin/env python3
""" This module defines the update_variables_momentum function. """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    This function updates a variable using the gradient descent with momentum
    optimization algorithm.
    Args:
        alpha (float): is the learning rate.
        beta1 (float): is the momentum weight.
        var (numpy.ndarray): is a numpy.ndarray containing the variable to be
        updated.
        grad (numpy.ndarray): is a numpy.ndarray containing the gradient of
        var.
        v (numpy.ndarray): is the previous first moment of var.
    Returns: the updated variable and the new moment, respectively.
    """
    # Compute the first moment of var using the gradient of var
    v = beta1 * v + (1 - beta1) * grad
    # Update the variable using the gradient descent with momentum algorithm
    var = var - alpha * v
    return var, v
