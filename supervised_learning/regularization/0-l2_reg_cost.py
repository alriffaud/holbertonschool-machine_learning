#!/usr/bin/env python3
""" This module defines the l2_reg_cost function. """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    This function calculates the cost of a neural network with L2
    regularization.
    Args:
        cost (numpy.ndarray): The cost of the network without L2
        regularization.
        lambtha (float): The regularization parameter.
        weights (dict): The weights and biases of the network.
        L (int): The number of layers in the network.
        m (int): The number of data points used.
    """
    l2_reg = 0
    for i in range(1, L + 1):
        l2_reg += np.linalg.norm(weights['W' + str(i)]) ** 2
    return cost + (lambtha / (2 * m)) * l2_reg
