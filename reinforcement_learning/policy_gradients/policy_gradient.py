#!/usr/bin/env python3
"""
Module for computing a simple policy function for reinforcement learning.
"""
import numpy as np


def policy(matrix, weight):
    """
    This function computes the policy given a state matrix and a weight matrix.
    The function applies a linear transformation on the state using the weight
    matrix and then applies the softmax function to obtain a probability
    distribution over actions.
    Args:
        matrix (np.ndarray): A 2D numpy array representing the state.
        weight (np.ndarray): A 2D numpy array representing the weight matrix.
    Returns:
        np.ndarray: A 2D numpy array containing the policy probabilities for
            each action.
    """
    # Compute the linear combination of the state and the weights
    z = np.dot(matrix, weight)
    # Apply the softmax function to obtain probabilities
    exp_z = np.exp(z)
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax
