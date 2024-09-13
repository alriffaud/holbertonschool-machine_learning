#!/usr/bin/env python3
""" This module defines the update_variables_Adam function. """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    This function updates a variable in place using the Adam optimization
    algorithm.
    Args:
        alpha (float): is the learning rate.
        beta1 (float): is the weight used for the first moment.
        beta2 (float):  is the weight used for the second moment.
        epsilon (float): is a small number to avoid division by zero.
        var (numpy.ndarray): is a numpy.ndarray containing the variable to be
        updated.
        grad (numpy.ndarray): is a numpy.ndarray containing the gradient of
        var.
        v (numpy.ndarray): is the previous first moment of var.
        s (numpy.ndarray): is the previous second moment of var.
        t (float): is the time step used for bias correction.
    Returns: the updated variable, the new first moment, and the new second
    moment, respectively.
    """
    # Compute the first moment of var using the gradient of var
    v = beta1 * v + (1 - beta1) * grad
    # Compute the second moment of var using the gradient of var
    s = beta2 * s + (1 - beta2) * np.square(grad)
    # Compute the bias-corrected first moment
    v_corrected = v / (1 - beta1 ** t)
    # Compute the bias-corrected second moment
    s_corrected = s / (1 - beta2 ** t)
    # Update the variable using the Adam optimization algorithm
    var = var - (alpha / (np.sqrt(s_corrected) + epsilon)) * v_corrected
    return var, v, s
