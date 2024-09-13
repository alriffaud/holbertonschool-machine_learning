#!/usr/bin/env python3
""" This module defines the update_variables_RMSProp function. """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    This function updates a variable using the RMSProp optimization algorithm.
    Args:
        alpha (float): is the learning rate.
        beta2 (float): is the RMSProp weight.
        epsilon (float): is a small number to avoid division by zero.
        var (numpy.ndarray): is a numpy.ndarray containing the variable to be
        updated.
        grad (numpy.ndarray): is a numpy.ndarray containing the gradient of
        var.
        s (numpy.ndarray): is the previous second moment of var.
    Returns: the updated variable and the new moment, respectively.
    """
    # Compute the second moment of var using the gradient of var
    s = beta2 * s + (1 - beta2) * np.square(grad)
    # Update the variable using the RMSProp algorithm
    var = var - (alpha / (np.sqrt(s) + epsilon)) * grad
    return var, s
